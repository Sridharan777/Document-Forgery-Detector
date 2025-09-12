# 🧾 Document Forgery Detection (Receipt)

 **AI-powered web app** that detects whether a receipt is **genuine** or **forged** using a deep learning model (ResNet50).  
Built with **PyTorch, Streamlit, and Grad-CAM** for full transparency and explainability.  

---

##  Features
✅ **AI Model:** ResNet50 CNN (high accuracy, trained on custom dataset)  
✅ **Explainable AI:** Grad-CAM heatmaps highlight which parts of the receipt the model used  
✅ **Confidence Gauge:** Shows how sure the model is (green = genuine, red = forged)  
✅ **Interactive UI:** Clean, minimal Streamlit interface with dark/light theme  
✅ **Deployable Anywhere:** One-click deployment on Streamlit Cloud  

---

##  Project Structure
```
📁 document-forgery-detection
 ┣ 📁 my_dataset/       # Training, validation, and test data (images + labels)
 ┣ 📁 models/           # Trained ResNet50 model (best_resnet50.pth)
 ┣ 📁 gradcam_outputs/  # Grad-CAM visualization outputs
 ┣ 📄 app.py            # Streamlit app (main code)
 ┣ 📄 requirements.txt  # Dependencies for deployment
 ┗ 📄 README.md         # Project overview (this file)
```

---

## Tech Stack
- **Deep Learning:** PyTorch + torchvision  
- **Model:** ResNet50 (transfer learning)  
- **Visualization:** Grad-CAM (manual hooks)  
- **Frontend:** Streamlit + Plotly gauge  
- **Deployment:** Streamlit Cloud  

---

##  How It Works
1.  **Upload a receipt image**  
2.  **Model predicts** whether it’s Genuine or Forged  
3.  **Confidence score** shows model’s certainty  
4.  **Grad-CAM heatmap** highlights important areas the model looked at  

---

##  Real-World Use Cases
-  **Banks & Finance:** Detect fake invoices or forged receipts in claims  
-  **Retail:** Prevent fraud in return/refund systems  
-  **Auditing:** Automate verification of large-scale financial transactions  

---

##  Results
-  **Validation Accuracy:** ~90% after class balancing  
-  **Better explainability** with Grad-CAM 

---

##  Author
**Sridharan M**    
 Email: *sridharan22092003@gamil.com.com*  
 LinkedIn: *https://www.linkedin.com/in/sridharan-m-155a29230/*  
