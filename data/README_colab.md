%cd data/
!wget http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz
! tar xvf hand_dataset.tar.gz
!python3 converter.py


%cd ..
!for img in data/images/test/*; do echo $img >> data/all_test.txt;done
!cat data/all_test.txt | head -n 30 > data/test.txt

# colab save weights to drive
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Create and upload a file.
uploaded = drive.CreateFile({'title': 'pretrained.pth'})
uploaded.SetContentFile('pretrained.pth')
uploaded.Upload()