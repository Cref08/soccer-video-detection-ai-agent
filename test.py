#!/usr/bin/env python3
"""
Test script for this ai agent.

Steps:
1. Load video (all frames or specified batch)
2. Run this ai agent.predict_batch
3. Visualize results with bounding boxes and keypoints
4. Save output video and frames
"""

from typing import List, Optional
from pathlib import Path

import numpy as np
import argparse
import json
import re
import sys
import cv2

# Add scorevision to path to import keypoints functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "soccer-video-detection-ai-agent"))

from ai_agent import AiAgent, TVFrameResult

def load_frames(video_path: Path, max_frames: int = None, start_frame: int = 0) -> List:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If max_frames not specified, load all frames
    if max_frames is None or max_frames <= 0:
        max_frames = total_frames - start_frame
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def _infer_so_py_tag(so_path: Path) -> str | None:
    """Extract CPython ABI tag like 'cp312' from a .so filename."""
    match = re.search(r"cpython-(\d)(\d+)", so_path.name)
    if not match:
        return None
    major = match.group(1)
    minor = match.group(2)
    return f"cp{major}{minor}"


def _current_py_tag() -> str:
    """Return current CPython tag like 'cp311'."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def visualize_results(
    frame: np.ndarray, 
    result: TVFrameResult,
    show_boxes: bool = True,
    show_keypoints: bool = True,
    show_warped_template: bool = True,
    template_alpha: float = 0.3,
    return_warped_template: bool = False
) -> np.ndarray:
    """
    Visualize detection results on a frame.
    
    Args:
        frame: Input frame
        result: TVFrameResult with boxes and keypoints
        show_boxes: Whether to draw bounding boxes
        show_keypoints: Whether to draw keypoints
        show_warped_template: Whether to overlay the warped floor template
        template_alpha: Alpha blending for warped template overlay (0-1)
        return_warped_template: If True, returns (vis_frame, warped_template) tuple
        
    Returns:
        Annotated frame, or (annotated_frame, warped_template) if return_warped_template=True
    """
    vis_frame = frame.copy()
    warped_template_output = None

    # Draw bounding boxes
    if show_boxes and result.boxes:
        for box in result.boxes:
            # Color by class
            # Note: cls_id can specify team in multiple ways:
            #   1. cls_id=6 (Team 1) or cls_id=7 (Team 2) - team encoded in class (from OSNet classification)
            #   2. cls_id=2 (Player) + team_id field - team specified separately
            colors = {
                0: (0, 255, 255),    # Ball - Yellow
                1: (255, 0, 255),    # Goalkeeper - Magenta
                3: (255, 255, 0),    # Referee - Cyan
                6: (0, 0, 255),      # Player Team 1 - Red
                7: (0, 255, 0),      # Player Team 2 - Green
                2: (255, 255, 255),  # Generic Player - White (will override below if team_id present)
            }
            
            # Label names by class
            label_names = {
                0: "Ball",
                1: "Goalkeeper",
                3: "Referee",
                6: "Team1",
                7: "Team2",
                2: "Player",  # Generic player (refined below with team_id)
            }
            
            # Determine final color and label based on cls_id
            if box.cls_id == 6:
                # Team 1 player (from OSNet/team classifier)
                color = (0, 0, 255)      # Red
                label_name = "Team1"
            elif box.cls_id == 7:
                # Team 2 player (from OSNet/team classifier)
                color = (0, 255, 0)      # Green
                label_name = "Team2"
            elif box.cls_id == 2:
                # Generic player - check if team_id field is present
                team_id = None
                if hasattr(box, 'team_id'):
                    team_id = box.team_id
                elif hasattr(box, 'team'):
                    team_id = box.team
                
                if team_id:
                    # Generic player with team info
                    team_str = str(team_id).strip().lower()
                    if team_str in {"1", "team1"}:
                        color = (0, 0, 255)      # Red for Team 1
                        label_name = "Team1"
                    elif team_str in {"2", "team2"}:
                        color = (0, 255, 0)      # Green for Team 2
                        label_name = "Team2"
                    else:
                        color = (255, 255, 255)  # White for unknown team
                        label_name = "Player"
                else:
                    # No team info
                    color = colors.get(box.cls_id, (255, 255, 255))
                    label_name = label_names.get(box.cls_id, "Player")
            else:
                # All other classes (Ball, Goalkeeper, Referee)
                color = colors.get(box.cls_id, (255, 255, 255))
                label_name = label_names.get(box.cls_id, f"C{box.cls_id}")
            
            # Draw box
            cv2.rectangle(vis_frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            
            # Draw label with confidence
            label = f"{label_name}:{box.conf:.2f}"
            cv2.putText(
                vis_frame, label, (box.x1, box.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
    idx_list = []        
    # Draw keypoints
    if show_keypoints and result.keypoints:
        for idx, (x, y) in enumerate(result.keypoints):
            if (x, y) != (0, 0):  # Only draw non-zero keypoints
                # Ensure coordinates are integers
                x, y = int(x), int(y)
                # Draw circle
                cv2.circle(vis_frame, (x, y), 6, (0, 255, 255), -1)
                cv2.circle(vis_frame, (x, y), 8, (255, 255, 255), 2)
                idx_list.append(idx)
                # Draw keypoint index
                cv2.putText(
                    vis_frame, str(idx + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
    
    # Add frame info
    info_text = f"Frame {result.frame_id} | Boxes:{len(result.boxes)} | KPs:{sum(1 for kp in result.keypoints if kp != (0, 0))}/32, {idx_list}"
    cv2.putText(
        vis_frame, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    
    if return_warped_template:
        return vis_frame, warped_template_output
    return vis_frame


def format_results_as_chute_output(results: List[TVFrameResult]) -> dict:
    """
    Format results to match chute template output format.
    
    Args:
        results: List of TVFrameResult objects
        
    Returns:
        Dictionary matching TVPredictOutput format
    """
    try:
        frame_results = []
        for frame_result in results:
            # Call model_dump() to match chute template
            frame_results.append(frame_result.model_dump())
        
        return {
            "success": True,
            "predictions": {"frames": frame_results},
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "predictions": None,
            "error": f"❌ There was a problem formatting results: {e}"
        }


def save_results(
    frames: List[np.ndarray],
    results: List[TVFrameResult],
    output_dir: Path,
    output_filename: str = "output_video.mp4",
    save_video: bool = True,
    save_json: bool = True,
    fps: float = 25.0,
    show_warped_template: bool = True,
    template_alpha: float = 0.3,
    save_warped_templates: bool = True
):
    """
    Save visualization results.
    
    Args:
        frames: Original frames
        results: Detection results
        output_dir: Output directory
        output_filename: Name of output video file
        save_video: Whether to save video
        save_json: Whether to save JSON results
        fps: Frames per second for output video
        show_warped_template: Whether to overlay warped floor template
        template_alpha: Alpha blending for template overlay
        save_warped_templates: Whether to save individual warped template images
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    
    # Save JSON results in chute format
    if save_json and results:
        json_path = output_dir / f"{Path(output_filename).stem}_results.json"
        formatted_output = format_results_as_chute_output(results)
        
        with open(json_path, 'w') as f:
            json.dump(formatted_output, f, indent=2)
        
        print(f"✅ JSON results saved: {json_path}")
        print(f"   Format: TVPredictOutput (matches chute template)")
        print(f"   Success: {formatted_output['success']}")
        
        # Handle both success and error cases
        predictions = formatted_output.get('predictions')
        if predictions and isinstance(predictions, dict):
            pred_frames = predictions.get('frames', [])
            print(f"   Frames: {len(pred_frames)}")
        elif formatted_output.get('error'):
            print(f"   Error: {formatted_output['error']}")
    
    # Create visualizations
    vis_frames = []
    warped_templates = []
    
    # Create directory for warped templates if needed
    if save_warped_templates and show_warped_template:
        warped_dir = output_dir / f"{Path(output_filename).stem}_warped_templates"
        warped_dir.mkdir(exist_ok=True)
    
    for frame, result in zip(frames, results):
        # Get visualization with optional warped template output
        vis_output = visualize_results(
            frame, 
            result, 
            show_warped_template=show_warped_template,
            template_alpha=template_alpha,
            return_warped_template=save_warped_templates and show_warped_template
        )
        
        # Handle return value (could be tuple or single frame)
        if isinstance(vis_output, tuple):
            vis_frame, warped_template = vis_output
            vis_frames.append(vis_frame)
            if warped_template is not None:
                warped_templates.append(warped_template)
                # Save individual warped template
                template_path = warped_dir / f"frame_{result.frame_id:06d}_warped.png"
                cv2.imwrite(str(template_path), warped_template)
        else:
            vis_frames.append(vis_output)
    
    # Report saved warped templates
    if save_warped_templates and show_warped_template and warped_templates:
        print(f"✅ Warped templates saved: {len(warped_templates)} images in {warped_dir}")
    
    # Create side-by-side comparison video if warped templates exist
    if save_video and warped_templates and len(warped_templates) == len(vis_frames):
        sidebyside_frames = []
        for vis_frame, warped_template in zip(vis_frames, warped_templates):
            # Resize warped template to match vis_frame height
            h, w = vis_frame.shape[:2]
            warped_resized = cv2.resize(warped_template, (w, h))
            # Concatenate horizontally
            sidebyside = np.hstack([vis_frame, warped_resized])
            sidebyside_frames.append(sidebyside)
        
        # Save side-by-side video
        sidebyside_path = output_dir / f"{Path(output_filename).stem}_sidebyside.mp4"
        h, w = sidebyside_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(sidebyside_path), fourcc, fps, (w, h))
        for frame in sidebyside_frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"✅ Side-by-side video saved: {sidebyside_path}")
    
    # Save video
    if save_video and vis_frames:
        video_path = output_dir / output_filename
        h, w = vis_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        
        print(f"\nSaving video: {video_path}")
        for vis_frame in vis_frames:
            video_writer.write(vis_frame)
        
        video_writer.release()
        print(f"✅ Video saved: {video_path}")
        print(f"   Frames: {len(vis_frames)}, Resolution: {w}x{h}, FPS: {fps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer Video Detection AI Agent test with visualization")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/root/soccer-video-detection-ai-agent/videos",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to single video file (optional, overrides --video-dir)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to process (0 = all frames)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=10,
        help="Frame offset passed to predict_batch",
    )
    parser.add_argument(
        "--n-keypoints",
        type=int,
        default=32,
        help="Number of keypoints to request",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save output video",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save JSON results in chute format",
    )
    parser.add_argument(
        "--show-warped-template",
        action="store_true",
        default=True,
        help="Overlay warped floor template on frames",
    )
    parser.add_argument(
        "--template-alpha",
        type=float,
        default=0.3,
        help="Alpha blending for warped template overlay (0.0-1.0)",
    )
    parser.add_argument(
        "--save-warped-templates",
        action="store_true",
        default=True,
        help="Save individual warped template images and side-by-side comparison video",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    
    # Get list of videos to process
    if args.video:
        # Single video mode
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_paths = [video_path]
    else:
        # Directory mode - process all .mp4 files
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        video_paths = sorted(video_dir.glob("*.mp4"))
        if not video_paths:
            raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")

    # Initialize AiAgent once
    hf_repo_path = Path(__file__).resolve().parent
    ai_agent = AiAgent(hf_repo_path)
    
    print(f"{'='*70}")
    print(f"Soccer Video Detection AI Agent - Batch Video Processing")
    print(f"{'='*70}")
    print(f"\nTotal videos to process: {len(video_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Process each video
    for video_idx, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*70}")
        print(f"VIDEO {video_idx}/{len(video_paths)}: {video_path.name}")
        print(f"{'='*70}")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Determine frames to process
        max_frames = args.frames if args.frames > 0 else total_frames - args.start_frame
        
        print(f"\nVideo: {video_path}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        print(f"\nProcessing:")
        print(f"  Start frame: {args.start_frame}")
        print(f"  Frames to process: {max_frames}")
        print(f"  Batch size: {args.batch_size}")

        # Warm-up (only for first video)
        if video_idx == 1:
            warmup_frames = load_frames(video_path, min(2, max_frames), args.start_frame)
            ai_agent.predict_batch(warmup_frames, args.offset, args.n_keypoints)

        # Process all frames in batches
        print(f"\n{'='*70}")
        print(f"PROCESSING ALL FRAMES")
        print(f"{'='*70}")
        
        all_frames = []
        all_results = []
        
        num_batches = (max_frames + args.batch_size - 1) // args.batch_size
        
        for batch_number in range(num_batches):
            batch_start = args.start_frame + batch_number * args.batch_size
            batch_size = min(args.batch_size, max_frames - batch_number * args.batch_size)
            frame_number = args.start_frame + batch_number * args.batch_size
            
            print(f"Predicting Batch: {batch_number + 1}/{num_batches}")
            
            batch_frames = load_frames(video_path, batch_size, batch_start)
            if not batch_frames:
                break
            
            batch_results = ai_agent.predict_batch(
                batch_images=batch_frames,
                offset=frame_number,
                n_keypoints=args.n_keypoints,
            )
            
            if batch_results:
                all_frames.extend(batch_frames)
                all_results.extend(batch_results)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE FOR {video_path.name}")
        print(f"{'='*70}")
        print(f"\nTotal frames processed: {len(all_results)}")
        if all_results:
            print(f"Total boxes detected: {sum(len(r.boxes) for r in all_results)}")
            valid_kps = [sum(1 for kp in r.keypoints if kp != (0, 0)) for r in all_results]
            print(f"Total valid keypoints: {sum(valid_kps)}")
            print(f"Average valid keypoints per frame: {np.mean(valid_kps):.2f}")
        
        # Save results with original video name
        if all_results and (args.save_video or args.save_json):
            save_results(
                all_frames, 
                all_results, 
                output_dir,
                output_filename=video_path.name,
                save_video=args.save_video,
                save_json=args.save_json,
                fps=fps,
                show_warped_template=args.show_warped_template,
                template_alpha=args.template_alpha,
                save_warped_templates=args.save_warped_templates
            )
        elif not all_results:
            print("⚠️  No results to save")
    
    print(f"\n{'='*70}")
    print(f"✅ ALL {len(video_paths)} VIDEOS PROCESSED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()