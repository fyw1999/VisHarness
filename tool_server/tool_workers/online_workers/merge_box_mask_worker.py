"""Concurrent CPU worker for merging box/mask results.

The merge algorithm and its response schema intentionally match the serial
implementation in ``merge_box_mask_worker_serial.py``.  The only scheduling
change is that synchronous CPU work is moved off FastAPI's event loop into a
bounded thread pool, so independent HTTP requests can run concurrently.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from email import message

import numpy as np
import re
import cv2
import uuid
import argparse
from PIL import ImageDraw, ImageFont
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.utils.utils import *
from pycocotools import mask as mask_utils
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None


def mask_edge_sides(mask, margin_ratio):
    """Return local image edges approached by a non-empty binary mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return frozenset()
    height, width = mask.shape
    ys, xs = np.where(mask)
    margin_x = max(2, int(round(width * margin_ratio)))
    margin_y = max(2, int(round(height * margin_ratio)))
    sides = set()
    if int(xs.min()) <= margin_x:
        sides.add("left")
    if int(xs.max()) >= width - 1 - margin_x:
        sides.add("right")
    if int(ys.min()) <= margin_y:
        sides.add("top")
    if int(ys.max()) >= height - 1 - margin_y:
        sides.add("bottom")
    return frozenset(sides)


def transformed_footprint(width, height, transform_to_img0):
    """Map an image rectangle into the axis-aligned img_0 footprint."""

    corners = np.asarray(
        [[0.0, 0.0, 1.0], [width, 0.0, 1.0], [0.0, height, 1.0], [width, height, 1.0]],
        dtype=float,
    )
    transformed = (np.asarray(transform_to_img0, dtype=float) @ corners.T).T
    return (
        float(transformed[:, 0].min()),
        float(transformed[:, 1].min()),
        float(transformed[:, 0].max()),
        float(transformed[:, 1].max()),
    )


def footprints_are_equal(first, second):
    """Return whether two sources cover the same img_0 region."""

    scale = max(1.0, *(abs(value) for value in (*first, *second)))
    return bool(np.allclose(first, second, rtol=0.0, atol=1e-6 * scale))


def _one_sided_boundary_witness(
    detection,
    other,
    intersection_bounds,
    original_size,
    margin_ratio,
):
    """Check whether their overlap lies near one internal edge of detection."""

    if not detection.get("has_split_ancestor", False):
        return False
    x0, y0, x1, y1 = detection["footprint"]
    ox0, oy0, ox1, oy1 = other["footprint"]
    ix0, iy0, ix1, iy1 = intersection_bounds
    original_width, original_height = original_size
    local_width = max(1, int(detection.get("local_width", 1)))
    local_height = max(1, int(detection.get("local_height", 1)))
    margin_x = max(2.0 * (x1 - x0) / local_width, (x1 - x0) * margin_ratio)
    margin_y = max(2.0 * (y1 - y0) / local_height, (y1 - y0) * margin_ratio)
    tolerance = 1e-6 * max(1.0, float(original_width), float(original_height))

    if (
        "left" in detection["edge_sides"]
        and x0 > tolerance
        and ox0 < x0 - tolerance
        and ox1 > x0 + tolerance
        and ix0 <= x0 + margin_x
    ):
        return True
    if (
        "right" in detection["edge_sides"]
        and x1 < original_width - tolerance
        and ox0 < x1 - tolerance
        and ox1 > x1 + tolerance
        and ix1 >= x1 - margin_x
    ):
        return True
    if (
        "top" in detection["edge_sides"]
        and y0 > tolerance
        and oy0 < y0 - tolerance
        and oy1 > y0 + tolerance
        and iy0 <= y0 + margin_y
    ):
        return True
    if (
        "bottom" in detection["edge_sides"]
        and y1 < original_height - tolerance
        and oy0 < y1 - tolerance
        and oy1 > y1 + tolerance
        and iy1 >= y1 - margin_y
    ):
        return True
    return False


def has_relevant_patch_boundary_evidence(
    first,
    second,
    intersection_bounds,
    original_size,
    margin_ratio,
):
    """Return whether mask overlap is consistent with a patch-cut fragment."""

    if not first.get("has_split_ancestor", False) or not second.get("has_split_ancestor", False):
        return False
    if footprints_are_equal(first["footprint"], second["footprint"]):
        return False
    return _one_sided_boundary_witness(
        first,
        second,
        intersection_bounds,
        original_size,
        margin_ratio,
    ) or _one_sided_boundary_witness(
        second,
        first,
        intersection_bounds,
        original_size,
        margin_ratio,
    )

class MergeBoxMask(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_name,
                 worker_addr = "auto",
                 no_register = False,
                 tool_name = "MergeBoxMask",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 iou_threshold = 0.8,
                 fragment_overlap_ratio = 0.2,
                 fragment_edge_margin_ratio = 0.03,
                 cpu_worker_threads = None,
                 ):
        self.iou_threshold = iou_threshold
        self.fragment_overlap_ratio = fragment_overlap_ratio
        self.fragment_edge_margin_ratio = fragment_edge_margin_ratio
        requested_workers = (
            limit_model_concurrency
            if cpu_worker_threads is None
            else cpu_worker_threads
        )
        self.cpu_worker_threads = min(
            max(1, int(requested_workers)),
            max(1, int(limit_model_concurrency)),
        )
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=self.cpu_worker_threads,
            thread_name_prefix="merge-box-mask",
        )
        super().__init__(
            controller_addr,
            worker_name,
            worker_addr,
            no_register,
            tool_name,
            limit_model_concurrency,
            host,
            port
            )

    def init_model(self):
        logger.info(f"No need to initialize model {self.tool_name}.")
        logger.info(
            "MergeBoxMask CPU worker threads: "
            f"{self.cpu_worker_threads} (request concurrency limit: "
            f"{self.limit_model_concurrency})"
        )
        self.model = None

    async def generate_gate_async(self, params):
        """Run the unchanged synchronous implementation outside the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu_executor,
            self.generate_gate,
            params,
        )

    def generate(self, params):
        ret = {"message": "", "status": ""}
        image_dict = params.get("image_dict", None)
        if image_dict is None:
            message = "No image_dict provided in parameters for MergeBoxMask tool."
            logger.error(message)
            ret["message"] = message
            ret["status"] = "error"
            return ret
        if "img_0" not in image_dict:
            message = "Original image 'img_0' is required in image_dict for merging, but not found."
            logger.error(message)
            ret["message"] = message
            ret["status"] = "error"
            return ret
        try:
            img_0_pil = bytes_to_pil(image_dict["img_0"]["image_bytes"])
            W_orig, H_orig = img_0_pil.size

            # --- 辅助函数：获取变换矩阵和图像的“层级深度” ---
            def get_transform_and_depth(img_name):
                metadata = image_dict[img_name]
                provided_transform = metadata.get("transform_to_img0")
                if provided_transform is not None:
                    transform = np.asarray(provided_transform, dtype=float)
                    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
                        raise ValueError(f"Invalid transform_to_img0 for image {img_name}")
                    return transform, int(metadata.get("depth", 0))

                T = np.eye(3)
                curr_name = img_name
                depth = 0

                while curr_name != "img_0":
                    if curr_name.endswith("_4x"):
                        T_step = np.array([[0.25, 0, 0],[0, 0.25, 0], [0, 0, 1]])
                        curr_name = curr_name[:-3]
                        depth += 1
                    else:
                        m = re.search(r'(.*)(_r\d+_c\d+)$', curr_name)
                        if m:
                            parent_name = m.group(1)
                            offset_x = image_dict[curr_name].get("offset_x", 0)
                            offset_y = image_dict[curr_name].get("offset_y", 0)

                            T_step = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
                            curr_name = parent_name
                            depth += 1
                        else:
                            raise ValueError(f"无法解析图像的继承关系: {curr_name}")
                    T = T_step @ T

                return T, depth

            # 1. 收集并映射所有的检测结果到 img_0 的全局坐标系
            global_detections =[]

            for img_name, data in image_dict.items():
                if "bboxes" in data and "masks" in data:
                    bboxes = np.array(data["bboxes"])
                    masks = np.transpose(mask_utils.decode(data["masks"]), (2, 0, 1))

                    if len(bboxes) == 0:
                        continue

                    T, depth = get_transform_and_depth(img_name)
                    M = T[:2, :]
                    local_height, local_width = masks.shape[1:]
                    footprint = transformed_footprint(local_width, local_height, T)
                    has_split_ancestor = bool(
                        data.get(
                            "has_split_ancestor",
                            re.search(r"_r\d+_c\d+(?:_|$)", img_name) is not None,
                        )
                    )

                    for i in range(len(bboxes)):
                        xmin, ymin, xmax, ymax = bboxes[i]
                        pt_min = T @ np.array([xmin, ymin, 1.0])
                        pt_max = T @ np.array([xmax, ymax, 1.0])

                        xmin_g = np.clip(pt_min[0], 0, W_orig)
                        xmax_g = np.clip(pt_max[0], 0, W_orig)
                        ymin_g = np.clip(pt_min[1], 0, H_orig)
                        ymax_g = np.clip(pt_max[1], 0, H_orig)

                        if xmin_g >= xmax_g or ymin_g >= ymax_g:
                            continue

                        mask = masks[i]
                        mask_bin = (mask > 0.5) if mask.dtype.kind in ('f', 'c') else (mask > 0)
                        mask_float = mask_bin.astype(np.float32)

                        # 恢复全局掩码
                        mask_g = cv2.warpAffine(mask_float, M, (W_orig, H_orig), flags=cv2.INTER_LINEAR) > 0.5
                        area = mask_g.sum()

                        if area == 0:
                            continue

                        global_detections.append({
                            "bbox":[xmin_g, ymin_g, xmax_g, ymax_g],
                            "mask": mask_g,
                            "source": img_name,
                            "depth": depth,
                            "area": area,
                            "footprint": footprint,
                            "has_split_ancestor": has_split_ancestor,
                            "edge_sides": mask_edge_sides(mask_bin, self.fragment_edge_margin_ratio),
                            "local_width": local_width,
                            "local_height": local_height,
                        })

            # 2. 层级 NMS (Hierarchical NMS)：剔除浅层母图的冗余检测
            global_detections.sort(key=lambda x: (x["depth"], x["area"]), reverse=True)

            kept_detections =[]

            def compute_mask_iou(det1, det2):
                b1, b2 = det1["bbox"], det2["bbox"]
                ixmin, iymin = max(b1[0], b2[0]), max(b1[1], b2[1])
                ixmax, iymax = min(b1[2], b2[2]), min(b1[3], b2[3])
                if ixmin >= ixmax or iymin >= iymax:
                    return 0.0

                ixmin_idx, iymin_idx = int(ixmin), int(iymin)
                ixmax_idx, iymax_idx = int(np.ceil(ixmax)), int(np.ceil(iymax))

                inter1 = det1["mask"][iymin_idx:iymax_idx, ixmin_idx:ixmax_idx]
                inter2 = det2["mask"][iymin_idx:iymax_idx, ixmin_idx:ixmax_idx]

                intersection = np.logical_and(inter1, inter2).sum()
                if intersection == 0:
                    return 0.0

                union = det1["area"] + det2["area"] - intersection
                return intersection / union if union > 0 else 0.0

            # 开始 NMS 剔除
            for det in global_detections:
                is_duplicate = False
                for kept_det in kept_detections:
                    if det["source"] == kept_det["source"]:
                        continue
                    iou = compute_mask_iou(det, kept_det)
                    if iou > self.iou_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    kept_detections.append(det)

            # 3. 基于掩码有效重叠性合并“被切开的物体碎片” [重点修改区域]
            N = len(kept_detections)
            candidate_edges = []

            for i in range(N):
                source_i = kept_detections[i]["source"]
                box_i = kept_detections[i]["bbox"]
                mask_i = kept_detections[i]["mask"]
                area_i = kept_detections[i]["area"]

                for j in range(i + 1, N):
                    source_j = kept_detections[j]["source"]

                    # 【新增规则1】：如果两个碎片来自同一张子图，说明模型已经判定它是俩独立物体，绝不合并！
                    if source_i == source_j:
                        continue

                    box_j = kept_detections[j]["bbox"]
                    ixmin, iymin = max(box_i[0], box_j[0]), max(box_i[1], box_j[1])
                    ixmax, iymax = min(box_i[2], box_j[2]), min(box_i[3], box_j[3])

                    # 如果外接矩形相交
                    if ixmin < ixmax and iymin < iymax:
                        mask_j = kept_detections[j]["mask"]
                        area_j = kept_detections[j]["area"]

                        ixmin_idx, iymin_idx = max(0, int(ixmin)), max(0, int(iymin))
                        ixmax_idx, iymax_idx = min(W_orig, int(np.ceil(ixmax))), min(H_orig, int(np.ceil(iymax)))

                        slice_i = mask_i[iymin_idx:iymax_idx, ixmin_idx:ixmax_idx]
                        slice_j = mask_j[iymin_idx:iymax_idx, ixmin_idx:ixmax_idx]

                        intersection_mask = np.logical_and(slice_i, slice_j)
                        intersection = intersection_mask.sum()

                        # 【新增规则2】：引入重叠比例判断，抛弃原先的 `.any()`
                        if intersection > 0:
                            min_area = min(area_i, area_j)
                            # The overlap must occupy a meaningful part of the
                            # smaller result and occur near a relevant internal
                            # patch boundary. This distinguishes crop fragments
                            # from nearby dense objects with incidental overlap.
                            if min_area > 0:
                                overlap_ratio = intersection / min_area
                                intersection_y, intersection_x = np.where(intersection_mask)
                                intersection_bounds = (
                                    float(ixmin_idx + intersection_x.min()),
                                    float(iymin_idx + intersection_y.min()),
                                    float(ixmin_idx + intersection_x.max()),
                                    float(iymin_idx + intersection_y.max()),
                                )
                                if (
                                    overlap_ratio > self.fragment_overlap_ratio
                                    and has_relevant_patch_boundary_evidence(
                                        kept_detections[i],
                                        kept_detections[j],
                                        intersection_bounds,
                                        (W_orig, H_orig),
                                        self.fragment_edge_margin_ratio,
                                    )
                                ):
                                    candidate_edges.append((float(overlap_ratio), i, j))

            # Merge strongest fragment correspondences first while preserving
            # the invariant that one final object contains at most one
            # detection from any source image.  Pairwise source checks alone
            # are insufficient: A1 and A2 from source A could otherwise be
            # connected transitively through one oversized detection B from
            # source B (A1 - B - A2).
            parent = list(range(N))
            component_sources = [{kept_detections[i]["source"]} for i in range(N)]

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union_if_source_disjoint(left, right):
                left_root, right_root = find(left), find(right)
                if left_root == right_root:
                    return
                if component_sources[left_root] & component_sources[right_root]:
                    return
                # Keep the smaller root for deterministic component ordering.
                if left_root > right_root:
                    left_root, right_root = right_root, left_root
                parent[right_root] = left_root
                component_sources[left_root].update(component_sources[right_root])

            for _, i, j in sorted(candidate_edges, key=lambda edge: (-edge[0], edge[1], edge[2])):
                union_if_source_disjoint(i, j)

            components_by_root = {}
            for i in range(N):
                components_by_root.setdefault(find(i), []).append(i)
            components = sorted(components_by_root.values(), key=lambda comp: comp[0])

            # 4. 生成合并后的最终结果
            final_bboxes = []
            final_masks =[]

            for comp in components:
                bboxes_comp = [kept_detections[idx]["bbox"] for idx in comp]
                xmin = min([b[0] for b in bboxes_comp])
                ymin = min([b[1] for b in bboxes_comp])
                xmax = max([b[2] for b in bboxes_comp])
                ymax = max([b[3] for b in bboxes_comp])
                final_bboxes.append([xmin, ymin, xmax, ymax])

                masks_comp = [kept_detections[idx]["mask"] for idx in comp]
                merged_mask = np.logical_or.reduce(masks_comp)
                final_masks.append(merged_mask)

            final_bboxes = np.array(final_bboxes) if final_bboxes else np.empty((0, 4))
            final_masks = np.array(final_masks) if final_masks else np.empty((0, H_orig, W_orig), dtype=bool)

            # 5. 在 img_0 上做可视化绘制
            vis_img_copy = img_0_pil.copy().convert("RGB")
            vis_arr = np.array(vis_img_copy)

            if len(final_bboxes) > 0:
                # A request-local legacy RNG preserves the exact colors produced
                # by np.random.seed(42) + np.random.randint while avoiding races
                # between concurrent requests through NumPy's global RNG state.
                random_state = np.random.RandomState(42)
                colors = random_state.randint(0, 255, (len(final_bboxes), 3), dtype=np.uint8)
                thickness = max(1, int(max(H_orig, W_orig) * 0.002))

                for i in range(len(final_bboxes)):
                    bbox = final_bboxes[i]
                    mask = final_masks[i]
                    color = colors[i]

                    roi = vis_arr[mask]
                    vis_arr[mask] = (roi * 0.5 + color * 0.5).astype(np.uint8)

                    cv2.rectangle(
                        vis_arr,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color.tolist(),
                        thickness
                    )

            vis_img_final = Image.fromarray(vis_arr)

            final_masks_hwn = np.transpose(final_masks, (1, 2, 0))
            final_masks_f = np.asfortranarray(final_masks_hwn.astype(np.uint8))
            rle_final_masks = mask_utils.encode(final_masks_f)
            merge_results = {
                "final_bboxes": final_bboxes.tolist(),
                "final_masks": rle_final_masks,
                "count": len(final_bboxes),
                "final_visual_image": pil_to_bytes(vis_img_final)
            }
            ret["results"] = merge_results
            ret["status"] = "success"
            ret["message"] = f"Tool MergeBoxMask executed successfully."
        except ValueError as e:
            logger.error(f"Error processing line parameters: {e}")
            raise
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, default="MergeBoxMask")
    parser.add_argument("--worker_name", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20037)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.8)
    parser.add_argument("--fragment-overlap-ratio", type=float, default=0.2)
    parser.add_argument("--fragment-edge-margin-ratio", type=float, default=0.03)
    parser.add_argument(
        "--cpu-worker-threads",
        type=int,
        default=8,
        help=(
            "Maximum number of MergeBoxMask requests that execute CPU work "
            "simultaneously. It is capped by --limit-model-concurrency."
        ),
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = MergeBoxMask(
        controller_addr=args.controller_address,
        worker_name = args.worker_name,
        worker_addr=args.worker_address,
        tool_name = args.tool_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        iou_threshold = args.iou_threshold,
        fragment_overlap_ratio = args.fragment_overlap_ratio,
        fragment_edge_margin_ratio = args.fragment_edge_margin_ratio,
        cpu_worker_threads = args.cpu_worker_threads,
    )
    worker.run()
