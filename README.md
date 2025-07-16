# AI-DRIVEN-FABRIC-DEFECT-DETECTUION-SYSTEM

Introduction Adidas, a global leader in sports apparel manufacturing, consistently seeks
innovative solutions to enhance product quality, minimize waste, and improve efficiency. As the
demand for high-quality sportswear increases, maintaining consistency in fabric quality
becomes increasingly challenging. To address these challenges, Adidas integrated Artificial
Intelligence (AI) technologies into its manufacturing processes, particularly focusing on fabric
defect detection. This case study explores the AI-driven approach developed by Adidas,
including the methodologies, technologies used, implementation, results, and future potential.
Problem Statement The sports apparel industry faces various challenges in maintaining quality
standards due to human error, inefficient manual inspections, high labor costs, and substantial
material wastage. Traditional fabric inspection processes are labor-intensive, inconsistent, and
prone to human fatigue, resulting in defective products entering the market. Adidas recognized
the need for an automated, accurate, and scalable system to improve quality control and reduce
resource wastage.
Solution Overview Adidas implemented an AI-Driven Fabric Defect Detection System utilizing
deep learning techniques, particularly Convolutional Neural Networks (CNNs). The approach
focuses on identifying fabric defects through image analysis, with the system capable of
distinguishing between defective and non-defective fabrics with high accuracy.
Process & Methodologies
1. Data Collection:
   
○ High-resolution images of various fabrics were collected, including both defective
and non-defective samples.

○ Types of defects considered include holes, stains, weaving errors, misalignment,
and scratches.

2. Data Preprocessing:

○ Images were resized, converted to grayscale, normalized, and augmented using
techniques such as rotation, flipping, zooming, and shifting to improve
robustness.

○ Data was split into training and testing sets to ensure effective model evaluation.

3. Model Architecture:
   
○ A Sequential CNN model was developed using TensorFlow and Keras.

○ The architecture consisted of convolutional layers, pooling layers, dropout layers,
and dense layers to extract features and classify images.

○ Class weights were adjusted to address the imbalance between defective and
non-defective samples.

4. Model Training:
   
○ The model was trained using binary cross-entropy loss and the Adam optimizer
with a learning rate of 0.001.

○ Training was performed over multiple epochs, with accuracy and loss metrics
tracked.

5. Model Evaluation:
    
○ Evaluation metrics included accuracy, precision, recall, F1-score, and confusion
matrix analysis.

○ The model's performance was validated against the test dataset, achieving high
detection accuracy.
