# 🖥️ Deploy to Server with port 8000

You can apply this application to Linux Server

## 📌Set up Database



## 📌Set up the Application on Server

**1. Open Terminal**

**2. Login as SuperUser and go to /home folder**

```bass
sudo su
cd ..
```

**3. Set Up Python environment, including Python, `pip`, and `virtualenv`**

```bass
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install virtualenv
```

**4. Clone project from repository github**

```bass
git clone https://github.com/vionaaindah/fraud_detection_system.git
```

**5. Go to directory project**

```bass
cd fraud_detection_system
```

**6. Create an isolated Python environment, and install django**

```bass
virtualenv env -p python3
source env/bin/activate
pip install django
```

**7. install dependencie and tensorflow**

```bass
pip install -r requirements.txt
```

**8. Configuring **`settings.py`****

- Open **`fraud_detection_system/settings.py`** for editing.

```bass
sudo vim fraud_detection_system/settings.py
```

<b>Note : </b> To **`editing file`**  type **`i`** and to **`save file`** type **`:wq`**


# [START db_setup]
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': '<nama_db>',
        'USER': '<user_db>',
        'PASSWORD': '<password_db>',
        'HOST': '<your_host>',
        'PORT': '5432',
    }
}
# [END db_setup]
```

Save **`settings.py`**


**9. Run the Django migrations to set up your models**

```bash
python manage.py makemigrations
python manage.py makemigrations api_detection
python manage.py migrate
```


**10. Create a screen and run the application**

- use screen so the application can always run in the background

```bash
screen
source env/bin/activate
python manage.py runserver 0.0.0.0:8000
```

<b>Note:</b> use **`screen -r`** to enter screen already exist and **`ctrl+a d`** to exit from screen