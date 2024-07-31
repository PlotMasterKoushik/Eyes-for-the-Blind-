import vart
import xir
import numpy as np
import cv2
import pyttsx3

engine = pyttsx3.init()
def load_model(modelpath):
    g = xir.Graph.deserialize(modelpath)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    return vart.Runner.create_runner(subgraphs[1], "run")

dpu_runner = load_model("yolov3_voc.xmodel")
def preprocess_frame(frame):
    frame = cv2.resize(frame, (416, 416))
    frame = frame.astype(np.float32)
    frame /= 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame


def detect_objects(frame):
    input_data = preprocess_frame(frame)
    input_tensor = dpu_runner.get_input_tensors()[0]
    output_tensor = dpu_runner.get_output_tensors()[0]

    input_data = {input_tensor.name: input_data}
    output_data = {output_tensor.name: np.empty(output_tensor.dims, dtype=np.float32)}

    job_id = dpu_runner.execute_async(input_data, output_data)
    dpu_runner.wait(job_id)

    return output_data[output_tensor.name]

def analyze_and_navigate(detections, frame):
    left_free = True
    right_free = True
    center_occupied = False

    frame_width = frame.shape[1]

    for detection in detections:
        x, y, width, height = detection['bbox']
        center_x = x + width / 2

        if center_x < frame_width / 3:
            right_free = False
        elif center_x > 2 * frame_width / 3:
            left_free = False
        else:
            center_occupied = True

    if center_occupied:
        if left_free:
            instruction = "Object detected. Move left."
        elif right_free:
            instruction = "Object detected. Move right."
        else:
            instruction = "Dead end"

    return instruction

def give_instruction(instruction):
    engine.say(instruction)
    engine.runAndWait()

def main():
    while True:
        frame = capture_frame()
        if frame is not None:
            detections = detect_objects(frame)
            instruction = analyze_and_navigate(detections, frame)
            print("Instruction:", instruction)
            give_instruction(instruction)

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

if __name__ == "__main__":
    main()
