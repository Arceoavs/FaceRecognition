import argparse
import os
import time

import cv2
import easygui
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import pairwise_distances

import detect_and_align


class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self,
        id_folder,
        mtcnn,
        sess,
        embeddings,
        images_placeholder,
        phase_train_placeholder,
        distance_treshold,
    ):
        print(f"Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        os.makedirs(id_folder, exist_ok=True)
        image_paths = [
            os.path.join(id_folder, id_name, img)
            for id_name in os.listdir(os.path.expanduser(id_folder))
            for img in os.listdir(os.path.join(id_folder, id_name))
        ]

        if not image_paths:
            return

        print(f"Found {len(image_paths)} images in id folder")
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {
            images_placeholder: aligned_images,
            phase_train_placeholder: False,
        }
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def add_id(self, embedding, new_id, face_patch):
        self.embeddings = (
            np.atleast_2d(embedding)
            if self.embeddings is None
            else np.vstack([self.embeddings, embedding])
        )
        self.id_names.append(new_id)

        id_folder = os.path.join(self.id_folder, new_id)
        os.makedirs(id_folder, exist_ok=True)

        numbered_filenames = [
            int(os.path.splitext(f)[0])
            for f in os.listdir(id_folder)
            if os.path.splitext(f)[0].isdigit()
        ]
        img_number = max(numbered_filenames, default=-1) + 1
        output_filename = os.path.join(id_folder, f"{img_number}.jpg")
        cv2.imwrite(output_filename, face_patch)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                warning_msg = (
                    "Warning: Found multiple faces in id image: %s\n"
                    "Make sure to only have one face in the id images. "
                    "If that's the case then it's a false positive detection and "
                    "you can solve it by increasing the thresholds of the cascade network"
                    % os.path.basename(image_path)
                )
                print(warning_msg)
            aligned_images.extend(face_patches)
            id_image_paths.extend([image_path] * len(face_patches))
            self.id_names += [os.path.basename(os.path.dirname(image_path))] * len(
                face_patches
            )

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        # Calculate the distance matrix between the input embeddings and the saved embeddings.
        distance_matrix = pairwise_distances(embs, self.embeddings)

        # Find the matching ID and distance for each row in the distance matrix.
        # Store the results as a list of tuples.
        # If self.id_names is None, set the ID to None and the distance to infinity for all rows.
        matching_results = (
            [
                (
                    self.id_names[np.argmin(distance_row)],
                    np.min(distance_row)
                    if np.min(distance_row) < self.distance_treshold
                    else np.inf,
                )
                for distance_row in distance_matrix
            ]
            if self.id_names
            else [(None, np.inf)] * len(embs)
        )

        # Unpack the results into two lists: matching_ids and matching_distances.
        # If self.id_names is None, return (None, [np.inf] * len(embs)).
        # Otherwise, return the two lists as a tuple.
        return zip(*matching_results) if self.id_names else (None, [np.inf] * len(embs))


def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    return cap


def load_model(model):
    """Load the model from the given path."""

    # Expand the user path and check if the given model is a file.
    model_exp = os.path.expanduser(model)
    if not os.path.isfile(model_exp):
        raise ValueError("Specify model file, not directory!")

    # Print the path of the model file being loaded.
    print(f"Loading model filename: {model_exp}")

    # Load the graph definition from the model file and import it.
    with tf.io.gfile.GFile(model_exp, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")


def process_frame(
    show_options,
    frame,
    mtcnn,
    sess,
    embeddings,
    images_placeholder,
    phase_train_placeholder,
    id_data,
):
    (
        face_patches,
        padded_bounding_boxes,
        landmarks,
    ) = detect_and_align.detect_faces(frame, mtcnn)

    if face_patches:
        face_patches = np.stack(face_patches)
        feed_dict = {
            images_placeholder: face_patches,
            phase_train_placeholder: False,
        }
        embs = sess.run(embeddings, feed_dict=feed_dict)

        matching_ids, matching_distances = id_data.find_matching_ids(embs)

        print("Matches in frame:")
        for bb, landmark, matching_id, dist in zip(
            padded_bounding_boxes,
            landmarks,
            matching_ids,
            matching_distances,
        ):
            matching_id = matching_id or "Unknown"
            print(f"Hi {matching_id}! Distance: {dist:.4f}")

            # Draw additional info on the frame.
            if show_options.get("id"):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    frame,
                    matching_id,
                    (bb[0], bb[3]),
                    font,
                    1,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if show_options.get("bb"):
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
            if show_options.get("landmarks"):
                for j in range(5):
                    size = 1
                    top_left = (
                        int(landmark[j]) - size,
                        int(landmark[j + 5]) - size,
                    )
                    bottom_right = (
                        int(landmark[j]) + size,
                        int(landmark[j + 5]) + size,
                    )
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
    else:
        print("Couldn't find a face")
    return


def handle_key_press(show_options, key, frame_detections, id_data):
    if key == ord("q"):
        return False
    show_options["landmarks"] ^= key == ord("l")
    show_options["bb"] ^= key == ord("b")
    show_options["id"] ^= key == ord("i")
    show_options["fps"] ^= key == ord("f")
    return (
        key != ord("s")
        or frame_detections is None
        or save_frame_detections(id_data, frame_detections)
        or True
    )


def save_frame_detections(id_data, frame_detections):
    for emb, bb in zip(frame_detections["embs"], frame_detections["bbs"]):
        patch = frame_detections["frame"][bb[1] : bb[3], bb[0] : bb[2], :]
        cv2.imshow("frame", patch)
        cv2.waitKey(1)
        new_id = easygui.enterbox("Who's in the image? Leave empty for non-valid")
        if len(new_id) > 0:
            id_data.add_id(emb, new_id, patch)


def main(args: argparse.Namespace) -> None:
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        # Setup models
        mtcnn = detect_and_align.create_mtcnn(sess, None)
        load_model(args.model)
        images_placeholder, embeddings, phase_train_placeholder = [
            tf.get_default_graph().get_tensor_by_name(name)
            for name in ["input:0", "embeddings:0", "phase_train:0"]
        ]

        # Load anchor IDs
        id_data = IdData(
            args.id_folder[0],
            mtcnn,
            sess,
            embeddings,
            images_placeholder,
            phase_train_placeholder,
            args.threshold,
        )

        # Setup camera
        cap = setup_camera()

        show_options = {"landmarks": False, "bb": False, "id": True, "fps": False}
        frame_detections = None

        while True:
            start = time.time()
            _, frame = cap.read()

            process_frame(
                show_options,
                frame,
                mtcnn,
                sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,
                id_data,
            )

            end = time.time()

            fps = round(1 / (end - start), 2) if show_options.get("fps") else None
            cv2.putText(
                frame,
                str(fps),
                (0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            ) if fps else None

            cv2.imshow("frame", frame)

            if not handle_key_press(
                show_options, cv2.waitKey(1), frame_detections, id_data
            ):
                break

        cap.release(), cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to model protobuf (.pb) file")
    parser.add_argument(
        "id_folder", type=str, nargs="+", help="Folder containing ID folders"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Distance threshold defining an id match",
        default=1.0,
    )
    main(parser.parse_args())
