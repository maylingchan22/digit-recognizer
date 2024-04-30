# digit-recognizer
 
### Project Topic:

The project focuses on digit recognition using the MNIST dataset, a well-known collection of handwritten images commonly used in computer vision tasks. The goal is to correctly identify digits from tens of thousands of these handwritten images. 

The type of learning algorithm used in this project is a Sequential Convolutional Neural Network (CNN), a type of deep learning algorithm commonly used for image classification tasks, such as recognizing digits in these handwritten images. CNNs are particularly well-suited for computer vision tasks because they can automatically learn to extract relevant features from images through the use of convolutional layers.

The specific task involved in this project is digit recognition, which is a classification problem. Given an input image of a handwritten digit, the algorithm must correctly classify it into one of ten possible classes (digits 0 through 9). This task involves identifying patterns and features within the images that distinguish one digit from another.

### Project Goal:

My goal for this project is to gain experience in exploring the real-world applications of digit recognition. By working with the MNIST dataset, I aim to accurately identifying handwritten digits images through techniques, such as CNNs models.

In the context of real-life application, such as optical character recognition systems, accurate digit recognition is essential for converting scanned documents or images containing numerical data into editable and searchable formats. This is broadly used in industries such as legal, healthcare, and education, where document management, archiving, and digitization efforts are prevalent.

In financial institutions, digit recognition is helpful in many operations, from processing checks and invoices to verifying credit card transactions and authenticating signatures. The ability to swiftly and accurately recognize digits is indispensable for automating routine tasks, detecting fraudulent activities, and ensuring compliance with regulatory requirements. Therefore, robust digit recognition systems contribute significantly to operational efficiency, risk mitigation, and customer trust in financial services.

### Data:

The dataset used in the project is sourced from Kaggle, specifically from the "Digit Recognizer" competition hosted on the platform. This dataset consists of tens of thousands of handwritten digit images, commonly known as the MNIST dataset, which has served as a benchmark for classification algorithms in computer vision since its release in 1999.

APA Style Citation:
AstroDave, Will Cukierski. (2012). Digit Recognizer. Kaggle. https://kaggle.com/competitions/digit-recognizer

### Data Description:

The training dataset consists of 42,000 samples, each containing 785 features. These features represent the pixel values of handwritten digit images, with one additional feature denoting the corresponding label for each image. Similarly, the testing dataset contains 28,000 samples, each with 784 features representing pixel values.

The pixel features in both the training and testing datasets are integers, indicating grayscale pixel values ranging from 0 to 255. Since the MNIST dataset comprises grayscale images, each pixel value represents the intensity of the corresponding pixel in the image. The dimensions of the images are standardized, with each image represented as a 28x28 grid of pixels flattened into a single row of 784 pixel features. This format ensures consistency across all samples and facilitates compatibility with algorithms.

### Exploratory Data Analysis:

The Exploratory Data Analysis (EDA) for the MNIST dataset reveals important insights into the characteristics of the handwritten digit images. Firstly, examining the distribution of labels shows that the dataset is relatively balanced, with each digit represented fairly evenly. For instance, digits 1, 7, and 3 have the highest counts with 4684, 4401, and 4351 samples respectively, while digit 5 has the lowest count with 3795 samples. This balanced distribution ensures that the model is exposed to sufficient samples of each digit during training, preventing bias towards any specific class.

Next, visualizing sample images for each digit provides a qualitative understanding of the dataset. The displayed images showcase the variability in handwriting styles across different digits, highlighting the challenges involved in accurately recognizing handwritten digits. For example, some digits like 1 and 7 exhibit clear and distinct shapes, while others like 4 and 9 may have more variability in their representations.

Furthermore, histograms of pixel intensity values provide insights into the distribution of pixel intensities across different digits. By binning pixel intensities into ranges and counting their occurrences, we can observe the distribution of lightness or darkness in the images. These histograms reveal that pixel intensities are spread across a wide range, indicating variations in brightness and contrast among the digit images.

Additionally, examining the histogram of image sizes reveals that the majority of images have a consistent size of 28x28 pixels, as indicated by the frequency count of 28. This consistency in image size simplifies preprocessing steps and ensures uniformity in input dimensions for later training models.

### Data Preprocessing

Data preprocessing is important for preparing the data in a suitable format for training models. For the MNIST dataset, which consists of handwritten digit images, several preprocessing techniques are applied to enhance the quality and diversity of the data. Firstly, the dataset is divided into features (pixel values) and labels (digit labels), allowing for separate processing of input data and target labels. This separation facilitates the application of specific preprocessing techniques tailored to each data type.

Another preprocessing step done is rescaling the pixel values to the range [0, 1] using Min-Max scaling. This normalization ensures that all pixel values are within the same range, which helps stabilize and accelerate the training process of deep learning models. Normalization also mitigates issues related to varying pixel intensity ranges across different images, ensuring consistent input data for the model.

Furthermore, the dataset is split into training and validation sets using the train_test_split function. This step helps evaluate the model's performance on unseen data and preventing overfitting. By reserving a portion of the data for validation, we can assess the model's generalization ability and adjust hyperparameters accordingly to improve performance.

Additionally, data augmentation techniques are employed to increase the diversity of the training data and improve the model's robustness. The ImageDataGenerator from Keras is utilized to apply various augmentation techniques such as rotation, shifting, shearing, and zooming to the images. These transformations create additional variations of the original images, which helps prevent overfitting and improves the model's ability to generalize to unseen data.

### Model

In the context of digit recognition from images, Convolutional Neural Networks (CNNs) are the standard choice due to their ability to effectively capture spatial dependencies in images. For this problem, I have chosen to implement a simple CNN architecture using Keras, which is well-suited for image classification tasks like MNIST digit recognition.

#### Model Architecture:
The CNN architecture consists of convolutional layers followed by max-pooling layers to extract features from the input images. The convolutional layers apply filters to the input images, capturing important patterns and features. The max-pooling layers reduce the spatial dimensions of the feature maps, making the model more computationally efficient and reducing overfitting. Finally, the output of the convolutional layers is flattened and passed through fully connected layers, leading to the final classification output.

#### Hyperparameters:
- Convolutional Layers: The number of convolutional layers and the number of filters in each layer are essential hyperparameters. Increasing the number of filters allows the model to capture more complex features, but also increases the model's computational complexity.
- Pooling Layers: The size of the pooling window affects the spatial dimensions of the feature maps. Smaller pooling windows lead to more aggressive down-sampling and can reduce overfitting, but may also discard valuable information.
- Dense Layers: The number of neurons in the fully connected layers determines the model's capacity to learn complex relationships in the data. Adding more neurons increases the model's capacity, but may also lead to overfitting if not regularized properly.
- Activation Functions: ReLU activation functions are commonly used in CNNs for their simplicity and effectiveness in alleviating the vanishing gradient problem.

#### Optimization and Training:
For optimization, I have chosen the Adam optimizer due to its adaptive learning rate properties and momentum updates, which help converge faster and avoid local minima. The choice of batch size and number of epochs also plays a crucial role in training. A larger batch size can lead to faster convergence but may require more memory, while training for too many epochs can lead to overfitting. I have set these hyperparameters initially as fairly small and then increase the hyperparameters numbers.

Overall, the chosen CNN architecture, along with the larger epoch hyperparameters (history 2) and optimization methods, is well-suited for the digit recognition task on the MNIST dataset with an ending accuracy of 0.9983. The architecture's ability to capture spatial dependencies in images, combined with the optimization method's efficiency, ensures effective training and accurate classification performance.

#### Results:

The analysis of the results from training the convolutional neural network (CNN) models for digit recognition on the MNIST dataset reveals several insights. Both models exhibited strong performance, achieving high accuracy and relatively low loss on both the training and validation datasets.

The first model, trained for 5 epochs with a batch size of 64, achieved a training accuracy of approximately 99.70% and a validation accuracy of around 99.12%. This model had a training loss of 0.0092 and a validation loss of 0.0380. On the other hand, the second model, trained for 10 epochs with a batch size of 128, attained a slightly higher training accuracy of 99.83% but a lower validation accuracy of 98.70%. The corresponding training and validation losses for this model were 0.0052 and 0.0492, respectively.

Visualizing the training and validation accuracy and loss using tables provides a clear overview of the model's performance across epochs. Both models demonstrate a gradual increase in accuracy and decrease in loss over epochs on the training dataset. However, there is a slight fluctuation in performance on the validation dataset, especially evident in the second model, where the validation loss increases after the 5th epoch.

Furthermore, the classification report generated for the second model provides detailed insights into the model's performance for each class. It includes precision, recall, and F1-score metrics, along with support values for each class. For instance, for digit 0, the model achieved a precision of 0.99, recall of 0.99, and F1-score of 0.99 with a support of 816 instances. Similar high precision, recall, and F1-scores were observed across all digits, indicating the model's ability to accurately classify digits in the MNIST dataset.

Overall, the chosen CNN architecture, along with the larger epoch hyperparameters (history 2) and optimization methods, proved to be well-suited for the digit recognition task on the MNIST dataset. The models exhibited strong performance, achieving high accuracy and effectively capturing spatial dependencies in images. 

#### Discussion and Conclusion

Upon reviewing the results and analysis, one aspect that warrants discussion is the observed fluctuation in validation accuracy during training, particularly in the second model. Despite achieving high training accuracy, the validation accuracy displayed some variability across epochs. This inconsistency raises questions about potential factors contributing to the model's performance instability.

One possible explanation for the fluctuation in validation accuracy could be the choice of hyperparameters or training methodology. While the model architecture itself may be well-suited for the task, variations in hyperparameters such as learning rate, batch size, or optimizer settings could impact the model's convergence and generalization ability. Additionally, the absence of techniques like dropout or batch normalization, which help regularize the model and improve stability, might contribute to the observed fluctuations.

Another factor that may have influenced the variability in validation accuracy is the complexity of the dataset itself. While MNIST is a relatively simple dataset with well-defined digits and uniform backgrounds, it is still subject to variations in digit appearance, noise, and other factors. These variations could pose challenges for the model, particularly during training, leading to fluctuations in performance as it learns to generalize across different digit representations.

Furthermore, the lack of extensive data augmentation techniques or regularization methods in the model training pipeline could also contribute to performance instability. Data augmentation techniques such as rotation, translation, and scaling can help expose the model to a more diverse range of digit variations, potentially improving its robustness and generalization ability. Similarly, regularization methods like dropout or weight decay can prevent overfitting and enhance the model's stability during training.

Reflecting on this project, I learned the significance of iterative experimentation and the value of interpreting model performance beyond mere accuracy metrics. Initially, my focus was primarily on maximizing accuracy, assuming it to be the sole indicator of model effectiveness. However, through this process, I gained a deeper understanding of the nuances involved in evaluating model performance comprehensively.

One key takeaway is the importance of considering additional evaluation metrics such as precision, recall, and F1-score. While accuracy provides a broad overview of the model's correctness, metrics like precision and recall offer insights into its performance across different classes. This project underscored the importance of balancing these metrics, especially in scenarios where class imbalances exist, as is often the case in real-world datasets.

Furthermore, I learned the significance of hyperparameter tuning and its impact on model convergence and generalization. Experimenting with different combinations of hyperparameters allowed me to observe their influence on training dynamics and performance stability. This iterative process of adjusting hyperparameters and analyzing their effects deepened my understanding of the interplay between model architecture, optimization methods, and training dynamics.

Looking ahead, there are several avenues for further exploration and improvement. Experimenting with different architectures, such as deeper or wider CNNs, could potentially enhance model performance by capturing more intricate patterns in the data. Additionally, exploring advanced techniques like transfer learning, where pre-trained models are adapted to new tasks, could provide insights into leveraging knowledge from larger datasets to improve performance on smaller ones like MNIST.
