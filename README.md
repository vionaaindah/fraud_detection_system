# fraud_detection_system
 API
### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/vionaaindah/fraud_detection_system.git
   cd fraud_detection_system
   ```
2. Install the required environment with virtualenv **(recommendation)**

   - Linux/macOs
     ```sh
     virtualenv env
     source env/bin/activate
     pip install -r requirements.txt
     ```
   - Windows
     ```sh
     python -m venv env
     env\scripts\activate
     pip install -r requirements.txt
     ```

3. Migrate the model to create databases

   ```sh
   python manage.py makemigrations
   python manage.py migrate
   ```

4. Start on the local webserver

   ```sh
   python manage.py runserver
   ```

   Now, your local webserver is running in `http://localhost:8000/`.