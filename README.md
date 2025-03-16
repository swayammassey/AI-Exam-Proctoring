AI Exam Proctoring System
📌 Overview
The AI Exam Proctoring System is designed to monitor students during online exams using AI-powered techniques such as face detection, eye tracking, and object detection. It helps ensure exam integrity by preventing cheating and unauthorized activities.

🚀 Features
🔍 Face Detection – Identifies the student appearing for the exam.
👀 Eye Tracking – Detects if the student is looking away from the screen.
📸 Screenshot Capture – Takes periodic snapshots for monitoring.
🎤 Audio Analysis – Detects background noises or multiple voices.
🛑 Cheating Alerts – Flags suspicious activities for review.
🛠️ Tech Stack
Python 🐍
OpenCV – Image processing
Mediapipe – Face and eye tracking
TensorFlow/PyTorch – AI models
Flask/Django (Optional) – Web-based proctoring system
🔧 Installation
Clone the Repository
bash
Copy
Edit
git clone https://github.com/swayammassey/AI-Exam-Proctoring.git
cd AI-Exam-Proctoring
Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🖥️ Usage
bash
Copy
Edit
python main.py
This will start the proctoring system and begin monitoring the exam environment.

📷 Sample Screenshots
Add some sample images or GIFs here to show how the system works.

📌 Future Enhancements
Live proctoring via a web dashboard
AI-based voice detection for speech monitoring
Automatic cheating report generation
🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

📜 License
This project is licensed under the MIT License.
