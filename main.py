import cv2
import os

def resize_and_maintain_aspect_ratio(image, target_height):
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

def compare_images(image1_path, image2_path, similarity=True):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: Could not load images.")
        return

    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate the similarity percentage
    similarity_percentage = len(matches) / max(len(keypoints1), len(keypoints2)) * 100
    print(f"The similarity percentage is: {similarity_percentage}" )
    # Draw the matches with lines connecting the keypoints
    matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if similarity:
        # Resize both images to have the same height while maintaining aspect ratio
        target_height = 800  # Set the target height for both images
        resized_image1 = resize_and_maintain_aspect_ratio(image1, target_height)
        resized_image2 = resize_and_maintain_aspect_ratio(image2, target_height)

        # Concatenate the images horizontally
        concatenated_image = cv2.hconcat([resized_image1, resized_image2])

        # Resize the matching result
        scaled_matching_result = cv2.resize(matching_result, (resized_image1.shape[1], target_height))

        # Display the concatenated image with matches and similarity percentage
        cv2.putText(scaled_matching_result, f"Similarity Percentage: {similarity_percentage:.2f}%",
                    (10, target_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Matching Keypoints", scaled_matching_result)
    else:
        # Resize the matching result
        screen_width, screen_height = 1920, 1080  # Change these values to match your screen resolution

        # Calculate the scale factor based on the width
        scale_factor = min(screen_width / matching_result.shape[1], screen_height / matching_result.shape[0])

        # Resize the matching result
        scaled_matching_result = cv2.resize(matching_result, None, fx=scale_factor, fy=scale_factor)

        # Display the matching result with similarity percentage
        cv2.putText(scaled_matching_result, f"Similarity Percentage: {similarity_percentage:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Matching Keypoints", scaled_matching_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image1_path = "dog_missing.png"
    directory = "./"  # Update the directory path to the folder containing "dog_random" images

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("dog_random"):
            image2_path = os.path.join(directory, filename)
            compare_images(image1_path, image2_path, similarity=True)
