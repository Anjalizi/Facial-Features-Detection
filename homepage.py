import cv2
import sys

print("Hi there! Welcome to Facial Features Detection Page." + "\n"+
	"Select a suitable option to proceed:" + "\n" + "\n" +
	"1. Face Detection" + "\n" +
	"2. Eye Detection" + "\n" +
	"3. Mouth Detection" + "\n"
	"Exit by pressing any other key." + "\n" )

choice = int(input("Enter your choice:"))

if(choice==1):
	# Get user supplied values
	imagePath = sys.argv[1]
	cascPath = "haarcascade_frontalface_default.xml"

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the image
	image = cv2.imread('o.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Faces found", image)
	cv2.waitKey(0)


if(choice==2):
	# Get user supplied values
	imagePath = sys.argv[1]
	cascPath = "haarcascade_eye.xml"

	# Create the haar cascade
	eyeCascade = cv2.CascadeClassifier(cascPath)

	image = cv2.imread('o.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect eyes in the image
	eyes = eyeCascade.detectMultiScale(
		gray,
		scaleFactor=2.27,
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("Found {0} eyes!".format(len(eyes)))

	# Draw a rectangle around the eyes
	for (x, y, w, h) in eyes:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Eyes found", image)
	cv2.waitKey(0)


if(choice==3):
	# Get user supplied values
	imagePath = sys.argv[1]
	cascPath = "haarcascade_smile.xml"

	# Create the haar cascade
	smileCascade = cv2.CascadeClassifier(cascPath)

	image = cv2.imread('o.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect smiles in the image
	smiles = smileCascade.detectMultiScale(
		gray,
		scaleFactor=3.4, #Increase the value for a greater accuracy
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("Found {0} smiles!".format(len(smiles)))

	# Draw a rectangle around the mouths
	for (x, y, w, h) in smiles:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Smiles found", image)
	cv2.waitKey(0)

else:
	sys.exit()
