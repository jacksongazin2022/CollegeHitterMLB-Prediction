# Does Tabular Data Augmentation Improve Predictive Performance for Major League Baseball Drafting?

**Authors**: Jackson Gazin and Anh Nguyen, Wake Forest University, USA

## Overview
This project explores whether augmenting tabular data improves the predictive performance of models tasked with forecasting which college baseball players will make it to Major League Baseball (MLB). The study leverages machine learning techniques, specifically focusing on handling class imbalance through data augmentation.

## Objectives
1. **Prediction Model**: Develop a classification model to predict if a college baseball player will reach MLB based on their performance metrics.
2. **Data Augmentation**: Use conditional tabular generative adversarial networks (CTGAN) to create synthetic data to address class imbalance and evaluate its impact on model performance.
3. **Evaluation**: Compare the performance of models trained on real data versus those trained on a combination of real and synthetic data.

## Methodology
- **Data Collection**: Data was gathered from the Chadwick Baseball Bureau, Baseball Reference, and WarrenNolan.com, focusing on college baseball players from 2012 to 2019. Metrics included College Weighted On-Base Average (WOBA), walk-to-strikeout ratios, stolen bases, age, and strength of schedule.
- **CRISP-DM Process**: The project followed the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology, encompassing business understanding, data understanding, data preparation, modeling, evaluation, and deployment.
- **Handling Class Imbalance**:
  - **Down-Sampling**: K-means clustering was used to balance the dataset by down-sampling the majority class.
  - **Up-Sampling**: CTGAN was employed to generate synthetic data, increasing the number of observations for the minority class.

## Key Findings
1. **Synthetic Data Impact**: The introduction of synthetic data did not improve the overall predictive performance of the model, as evidenced by lower specificity and marginally higher sensitivity.
2. **Best Model**: The optimal model was a logistic regression using only real data, achieving an accuracy of 0.7501. It correctly predicted 85.57% of players who made it to the MLB and 78.22% of those who did not.
3. **Model Comparison**: Despite extensive testing with different algorithms (Decision Trees, Random Forest, XGBoost, Support Vector Machines, and k-Nearest Neighbors), none significantly outperformed the baseline logistic regression model on the down-sampled dataset.

## Conclusion
The study concluded that while data augmentation using CTGAN provided a balanced dataset, it did not enhance the predictive performance of the models. The logistic regression model trained on down-sampled real data remained the best performing model, highlighting the challenges and limitations of synthetic data in improving predictive accuracy for this specific application.

## Future Work
Future research could explore alternative methods of data augmentation and the integration of additional qualitative variables such as scouting reports and physical performance metrics. Additionally, updating the model annually with new data could further refine its predictive capabilities.

## Acknowledgments
The authors extend their gratitude to Dr. Natalia Khuri for her invaluable feedback and comments. Additionally, the authors used ChatGPT for spelling checks of the document.

## Data and Code Availability
All data and code used in this study are available on [GitHub]([https://github.com/](https://github.com/jacksongazin2022/CollegeHitterMLB-Prediction/tree/main)).

## References
- Baumer, B. S., & Albert, J. (2018). Analyzing Baseball Data with R. Chapman and Hall/CRC.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In KDD '16.
- Danovitch, J. (2019). Trouble with the Curve: Predicting Future MLB Players Using Scouting Reports.
- Gerber, E. A. E., & Craig, B. A. (2021). A mixed effects multinomial logistic-normal model for forecasting baseball performance. Journal of Quantitative Analysis in Sports.
- Li, D.-C., Chen, S.-C., Lin, Y.-S., & Huang, K.-C. (2021). A Generative Adversarial Network Structure for Learning with Small Numerical Data Sets. Applied Sciences.
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling Tabular data using Conditional GAN. NeurIPS.
