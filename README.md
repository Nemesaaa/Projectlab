# Iterative GNN-based traffic flow prediction and sensor placement optimization

Abstract
Accurate urban road traffic flow prediction is crucial for intelligent transportation systems (ITS). This project presents a co-design approach to the Traffic Sensor Location Problem (TSLP) and traffic flow prediction using Graph Neural Networks (GNNs).

## Key Contributions:
- **Optimized Sensor Placement:** The city is divided into manageable districts to ensure that sensors capture representative traffic data from critical zones.  
- **GNN-Based Prediction:** The collected sensor data is processed using GNNs to improve traffic flow prediction accuracy.  
- **Efficient Cost-Accuracy Tradeoff:** The method balances **prediction accuracy** with **cost-effective sensor coverage**, ensuring high performance even with **only 10% of roads covered by sensors**.  
- **Validation:** The approach was tested using the **SUMO traffic simulator** on the road network of **Győr, Hungary**, demonstrating robust estimation and prediction capabilities.  


## Repository Structure:
• **`codes/`** – Contains the core logic for building the model, along with tools for visualizing the results.  
• **`data/`** – Includes all input files required to run the model.  
• **`forecasting/`** – Stores the results of the forecasting model and the corresponding visualization scripts.  
• **`generate/`** – Contains the generated SUMO outputs used in the study.  
• **`gnn_outputs/`** – Holds the results of the current-time estimation model, along with visualization scripts.  
• **`model_modification/`** – Includes experiments with different hidden dimensions and comparative graphs.  
• **`Zone_Based_Algorithm`** – Contains the core logic for the algorithm we developed.  

