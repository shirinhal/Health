
# Clear the environment
rm(list = ls())

# Load necessary libraries
library(ggplot2)
library(caret)
library(corrplot)
library(GGally)
library(ggcorrplot)
library(psych)
library(dplyr)
library(car)          # For VIF calculation
library(nnet)         # For multinomial logistic regression
library(vcd)          # For Cramér's V calculation
library(reshape2)     # For reshaping data in the ANOVA p-value heatmap
library(DMwR2)        # For SMOTE
library(randomForest) # For Random Forest
library(gbm)          # For Gradient Boosting
library(pROC)         # For ROC analysis

set.seed(123)

# Set working directory
setwd("~/Desktop/GTX/MGT6203/Health_MGT6203_Project")

# Load the dataset
file_path <- "./health_data.csv"
data <- read.csv(file_path)

# Remove unnecessary columns, including GeneticRisk
data <- data %>% dplyr::select(-PatientID, -GeneticRisk)

# Split BloodPressure into Systolic and Diastolic Pressure
data <- data %>%
  mutate(SystolicPressure = as.numeric(sub("/.*", "", BloodPressure)),
         DiastolicPressure = as.numeric(sub(".*/", "", BloodPressure))) %>%
  dplyr::select(-BloodPressure)

# Convert AnnualCheckups to a factor with reference level 4
data$AnnualCheckups <- relevel(as.factor(data$AnnualCheckups), ref = "4")

# Convert categorical variables to factors
categorical_vars <- c("Gender", "SmokingStatus", "AlcoholConsumption", "ExerciseFrequency", 
                      "Diabetes", "HeartDisease", "PhysicalActivityLevel", 
                      "DietQuality", "MedicationAdherence", "Outcome")
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Original Analysis: Count unique items in each category for categorical columns
category_counts <- lapply(categorical_vars, function(col) {
  as.data.frame(table(data[[col]]))
})
names(category_counts) <- categorical_vars
category_counts

# Save category counts to individual CSV files
for (col_name in names(category_counts)) {
  write.csv(category_counts[[col_name]], paste0(col_name, "_counts.csv"), row.names = FALSE)
}

# Create and display separate bar plots for each categorical variable
for (col in categorical_vars) {
  plot <- ggplot(data, aes_string(x = col)) +
    geom_bar(fill = "lightblue", color = "black") +
    labs(title = paste("Distribution of", col), x = col, y = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10))
  
  # Print the plot for the current column
  print(plot)
}

# Check and visualize missing values
missing_data <- colSums(is.na(data))
print(missing_data)

# --------------------------------------------
# Original Analysis: Chi-square Test
# --------------------------------------------
for (col in categorical_vars[-which(categorical_vars == "Outcome")]) {
  table_data <- table(data[[col]], data$Outcome)
  chi_test <- chisq.test(table_data, simulate.p.value = TRUE, B = 10000)
  cat("Variable:", col, "
")
  print(chi_test)
  cat("
")
}

# Original Analysis: Structure, Summary, and Outliers
str(data)
head(data)
summary(data)
describe(data)

# Outlier Detection Using Z-Score Method
numeric_columns <- sapply(data, is.numeric)
z_scores <- as.data.frame(scale(data[, numeric_columns]))
outlier_counts <- sapply(z_scores, function(x) sum(abs(x) > 3))
print("Outlier Counts per Column:")
print(outlier_counts)

# Apply Yeo-Johnson Transformation for Outlier Columns
data$CholesterolLevel_YeoJohnson <- yjPower(data$CholesterolLevel + 1, 
                                            powerTransform(data$CholesterolLevel + 1, family = "yjPower")$lambda)
data$HealthcareCost_YeoJohnson <- yjPower(data$HealthcareCost + 1, 
                                          powerTransform(data$HealthcareCost + 1, family = "yjPower")$lambda)

# Remove unnecessary columns
data <- data %>% dplyr::select(-CholesterolLevel, -HealthcareCost)

# Feature Engineering - Creating New Variables
data <- data %>%
  mutate(HealthRiskCategory = case_when(
    Age >= 65 & BMI >= 30 & CholesterolLevel >= 240 ~ "High Risk",
    Age >= 45 & (BMI >= 25 | CholesterolLevel >= 200) ~ "Moderate Risk",
    TRUE ~ "Low Risk"
  ))

data <- data %>%
  mutate(Diabetes_HeartDisease = interaction(Diabetes, HeartDisease),
         BMI_PhysicalActivity = BMI * as.numeric(PhysicalActivityLevel))

# Recursive Feature Elimination for Feature Selection
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
outcome <- data$Outcome
predictors <- data %>% dplyr::select(-Outcome)

# Apply Recursive Feature Elimination (RFE)
rfe_results <- rfe(predictors, outcome, sizes = c(1:10), rfeControl = control)
selected_features <- predictors[, predictors %in% predictors(rfe_results)]
data_selected <- cbind(outcome, selected_features)

# Standardize numerical columns after transformations
num_cols <- sapply(data, is.numeric)
data_scaled <- data
data_scaled[, num_cols] <- scale(data[, num_cols])

# Split the refined dataset into Training, Validation, and Test sets
train_index <- createDataPartition(data_selected$outcome, p = 0.7, list = FALSE)
train_data <- data_selected[train_index, ]
test_val_data <- data_selected[-train_index, ]
val_index <- createDataPartition(test_val_data$outcome, p = 0.5, list = FALSE)
val_data <- test_val_data[val_index, ]
test_data <- test_val_data[-val_index, ]

# Hybrid SMOTE with Downsampling
train_data_smote <- SMOTE(Outcome ~ ., data = train_data, perc.over = 100, perc.under = 200)
print("Class distribution after SMOTE and downsampling:")
print(table(train_data_smote$Outcome))

# Original Analysis: Multinomial Logistic Regression
multinom_model <- multinom(Outcome ~ ., data = data_scaled)
summary(multinom_model)

# Filter out low-impact variables
selected_vars <- c("Outcome", "Age", "CholesterolLevel_YeoJohnson", "HealthcareCost_YeoJohnson",
                   "ExerciseFrequency", "Diabetes", "HeartDisease")
data_refined <- data_scaled[, selected_vars]

# Model fitting on refined data
control <- trainControl(method = "cv", number = 10)
multinom_refined_model <- train(Outcome ~ ., data = train_data, method = "multinom", trControl = control)
val_predictions <- predict(multinom_refined_model, newdata = val_data)
val_accuracy <- mean(val_predictions == val_data$Outcome)
cat("Validation Accuracy of Refined Model:", val_accuracy, "\n")

# Gradient Boosting Model with Cross-Validation
control <- trainControl(method = "cv", number = 10, sampling = "smote")
gbm_model <- train(
  Outcome ~ ., data = train_data_smote, method = "gbm",
  trControl = control, verbose = FALSE, tuneLength = 5
)

# Display model results
print(gbm_model)

# ROC-AUC Evaluation on Test Set
pred_probs <- predict(gbm_model, newdata = test_data, type = "prob")
roc_curve <- multiclass.roc(test_data$outcome, pred_probs)
print(roc_curve)

# Feature Importance in Gradient Boosting Model
print("Feature Importance in Gradient Boosting Model:")
print(summary(gbm_model))
