from sklearn.model_selection import train_test_split


# Main function that will split the dataset
# Try to pass numpy arrays for images and labels
# @param: Images size of individual images should be consistent

def split_dataset(images, labels, train_split, shuffle):
    train_images, validation_images, train_labels, validation_labels = train_test_split(images,
                                                                                        labels,
                                                                                        train_size= train_split,
                                                                                        stratify=labels,
                                                                                        shuffle= shuffle)
    print(f'Train shape: {train_images.shape}')
    print(f'Train labels: {train_labels.shape}')
    print(f'Validation shape: {validation_images.shape}')
    print(f'validation labels: {validation_labels.shape}')

    return train_images, validation_images, train_labels, validation_labels

# Example use case
# train_x, val_x, train_y, val_y = split_dataset(images=img,
#                                                 labels=label,
#                                                 train_split=0.75,
#                                                 shuffle=True)
