#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <fstream>
#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

// Dupanloup Remy
// Projet Vision par Ordinateur
// Etape 3
// Mars 2015

struct CamShiftData
{
    Rect trackWindow;
    int hsize;
    float hranges[2];
    const float* phranges;
    Mat hsv, hue, mask, hist, backproj;
    int trackObject;
    int vmin, vmax, smin;
    int ch[2];
};

// Affiche ce que fais le code, au début de l'éxécution
static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

// Déclaration de detectAndDraw
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip, Rect& previousRoi, Rect roi, CamShiftData& );

// Déclaratio de myCamShift
Rect myCamShift( Mat&, Rect, CamShiftData&);

string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

// Programme principal
int main( int argc, const char** argv )
{
    help();

    CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    // Initialisation de la reconnaissance de la lettre
    CvANN_MLP mlp;
    const int class_count = 26;
    mlp.load("./neurones.xml");
    printf("j'ai loader le fichier");
    
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();

    string inputName;
    bool tryflip = false;
    
    // Les rectangles encadrant la zone de recherche
    Rect roi, previousRoi; 

    // On initialise les valeurs necessaire pour myCamShift
    CamShiftData camShiftData;
    camShiftData.hsize = 16;
    camShiftData.hranges[0] = 0;
    camShiftData.hranges[1] = 180;
    camShiftData.phranges = camShiftData.hranges;
    camShiftData.trackObject = 0;
    camShiftData.vmin = 10;
    camShiftData.vmax = 256;
    camShiftData.smin = 30;
    camShiftData.ch[0] = 0;
    camShiftData.ch[1] = 0;

    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    // Gestion des éléments passé en paramètre et des éventuelles erreurs
    for( int i = 1; i < argc; i++ )
        {
            cout << "Processing " << i << " " <<  argv[i] << endl;
            if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
            {
                if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                    scale = 1;
                cout << " from which we read scale = " << scale << endl;
            }
            else if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
            {
                tryflip = true;
                cout << " will try to flip image horizontally to detect assymetric objects\n";
            }
            else if( argv[i][0] == '-' )
            {
                cerr << "WARNING: Unknown option %s" << argv[i] << endl;
            }
            else
                inputName.assign( argv[i] );
        }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' ); // correct 0
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }

    cvNamedWindow( "result", 1 );

    // Boucle de détection des visages 
    if( capture )
    {
        cout << "In capture ..." << endl;
        int compteur = 1;
        for(;;)
        {
        	// Convertion de la video en image
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;

            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            // La première zone de recherche est de la taille de l'image
            roi = Rect(0,0,frameCopy.cols, frameCopy.rows);            
            
            // Si un objet est tracqué, on reste dans camShift
            if(camShiftData.trackObject){
            	Rect hand;
            	hand = myCamShift(frameCopy, previousRoi, camShiftData);
                
                char c = (char)waitKey(20);
                if ( c == 27 )
                {
                    goto _cleanup_;
                }
                if (c >= 0)
                {
                    Mat ret = camShiftData.backproj(hand);              
                    resize(ret, ret, Size(16,16), 0, 0, INTER_LINEAR);                    
                    Mat img = ret.reshape(0,1);

                    if ( c >= 'a' && c <= 'z'){ // On enregistre l'image de la main avec la lettre associé

                    	printf("j'ai appuyer sur une lettre !\n");
                        imshow("new hand", ret);                        
                        stringstream stream;
                        stream << img;

                        string image = stream.str();
                        image.erase(remove(image.begin(), image.end(), ' '), image.end());
                        image.erase(remove(image.begin(), image.end(), '['), image.end());
                        image.erase(remove(image.begin(), image.end(), ']'), image.end());

        				ofstream os("image.txt", ios::out | ios::app);
        				os << (char) toupper(c) << ",";
        				os << image << std::endl;
        				os.close();

				    } else if (c == 32) { // Affichage de la lettre après appuye sur espace

                        printf("j'ai appuyé sur espace !\n");
                        img.convertTo(img, CV_32FC1);
                        CvMat img2 = img;
                        CvMat* mlp_response = cvCreateMat( img2.rows, class_count, CV_32F );
                        CvPoint max_loc = {0,0};
                        mlp.predict(&img2, mlp_response);
                        cvMinMaxLoc(mlp_response, 0, 0, 0, &max_loc, 0);
                        int best_class = max_loc.x + 'A';
                        printf("la lettre supposée est : %c\n", ((char) best_class));
                    }
                }
            }

            else {            	
                detectAndDraw( frameCopy, cascade, nestedCascade, scale, tryflip, previousRoi, roi, camShiftData );
            }
        }
		_cleanup_:
        cvReleaseCapture( &capture );
    }

    cvDestroyWindow("result");

    return 0;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip, Rect& previousRoi, Rect roi, CamShiftData& camShiftData )
{
	int i = 0;
    double t = 0;

    // Listes contenant les rectangles encadrant les visages
    vector<Rect> faces, faces2;

    // Tableau de couleurs
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)
    };

    // Définition d'une matrice, qui contiendra l'image en gris dans laquelle on travaillera               
    Mat gray;

    // On sélectionne une petite zone de l'image dans laquelle on travaillera.
    Mat smallImg( cvRound (roi.height/scale), cvRound(roi.width/scale), CV_8UC1 );
    
    // La matrice contenant l'image dans le roi
    Mat matRoi = img(roi);

    // Onconvertit l'image de matRoi en niveau de gris
    cvtColor( matRoi, gray, CV_BGR2GRAY );

    // Redimentionnement de l'immage avec la taille de smallImg
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    
    // Optimisation du contraste
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();

    // Algorithm de détection du visage, stocké dans faces
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );

    // On recherche des visages également sur une image inversée, stocké dans faces2
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );

        // On replace les visages trouvé dans faces.
        if(!faces2.empty())
       	{
       		// Ajout du visage dans faces
       	    Rect visage = faces2[0]; 
       	    faces.push_back(Rect(smallImg.cols - visage.x - visage.width, visage.y, visage.width, visage.height));
       	}
    }

    // Mesure du temps d'éxécution
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    
    // Si aucun visage n'est trouvé, on cherchera pour l'itération suivante dans l'image entière
    if(faces.empty())
    {
    	// la zone de recherche fais la taille de l'image
    	previousRoi = Rect( 0, 0, img.cols, img.rows );

    	// Dessin du rectangle    	
    	rectangle( img, previousRoi, colors[3], 2, 8, 0);
    }

    // Si un visage est trouvé :
    else
    {   
    	//  On sélectionne le premier    
	    Rect* r = &faces[0];
	    Point center;	    
	    int radius;
	    camShiftData.trackObject = -1;

	    /* Le centre du cercle entourant le visage est à repositionner
	       dans l'image entière */
	    center.x = cvRound(r->x * scale + (r->width * 0.5 * scale) + roi.x);
	    center.y = cvRound(r->y * scale + (r->height * 0.5 * scale) + roi.y);
	    radius = cvRound((r->width + r->height)*0.25*scale);
	    
	    // Dessin du cercle
	    circle( img, center, radius, colors[0], 3, 8, 0 );
	    
	    
	    /* Définition des paramètre du rectangle previousRoi de la future
	       zone de recherche, par rapport au cercle entourant le visage*/
	    int x = center.x - radius - 50;
	    int y = center.y - radius - 50;
	    int width = radius * 2 + 100;
	    int height = radius * 2 + 100;
	    
	    // Vérification que le rectangle ne sorte pas de l'image
	   	if ( x <= 0 ) x = 0;
	   	if ( y <= 0) y = 0;
	 	if ( width + x > img.cols ) width = img.cols - x;
	 	if ( height + y > img.rows ) height = img.rows - y;
	 	
	 	// Création du rectangle
	 	previousRoi = Rect( x, y, width, height );

	 	// Dessin du rectangle
	 	rectangle( img, previousRoi, colors[3], 2, 8, 0);	       	
    }
    
    // Affichage de l'image    
	imshow( "result", img );
}

Rect myCamShift(Mat& img, Rect select, CamShiftData& camShiftData){
	// Convertion de l'image en couleur en hsv
	cvtColor(img, camShiftData.hsv, COLOR_BGR2HSV);


	int _vmin = camShiftData.vmin, _vmax = camShiftData.vmax;

    inRange(camShiftData.hsv, Scalar(0, camShiftData.smin, MIN(_vmin, _vmax)),
  			Scalar(180, 256, MAX(_vmin, _vmax)), camShiftData.mask);
    camShiftData.hue.create(camShiftData.hsv.size(), camShiftData.hsv.depth());
    mixChannels(&camShiftData.hsv, 1, &camShiftData.hue, 1, camShiftData.ch, 1);

    if( camShiftData.trackObject < 0 )
    {
    	printf("initialisation histogramme\n"); // On initialise l'histogramme une seul fois après avoir detecter un premier visage
    	Mat roi(camShiftData.hue, select), maskroi(camShiftData.mask, select);
        calcHist(&roi, 1, 0, maskroi, camShiftData.hist, 1, &camShiftData.hsize, &camShiftData.phranges);
        normalize(camShiftData.hist, camShiftData.hist, 0, 255, CV_MINMAX);

        // la trackWindow devient le rectangle de detection du visage
        camShiftData.trackWindow = select;
        camShiftData.trackObject = 1;
    } 

    // Premier appel du suivi avec la methode CamShift
    calcBackProject(&camShiftData.hue, 1, 0, camShiftData.hist, camShiftData.backproj, &camShiftData.phranges);
    camShiftData.backproj &= camShiftData.mask;
    RotatedRect trackBox = CamShift(camShiftData.backproj, camShiftData.trackWindow,
    					TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
    
    if( camShiftData.trackWindow.area() <= 1 )
    {
    	camShiftData.trackObject = 0;
    }

    // On dessine un carre noir par dessus le backproj autour de la premiere zone de recherche qui est le visage
    Rect dessinNoir = trackBox.boundingRect();
    
    // On l'agrandi un peu pour etre sur d'avoir tout le visage
    dessinNoir.x -= 30; // Pour cacher aussi le sommet de la tete
    dessinNoir.y -= 20;
    dessinNoir.width += 20;
    dessinNoir.height += 50; // Une valeur un peu plus grande pour cacher le coup

    // le trackwindows peut sortir de l'image, on verifie pour le dessin que ce ne soit pas le cas
    if (dessinNoir.x < 0) dessinNoir.x = 0;
    if (dessinNoir.y < 0) dessinNoir.y = 0;
    if (dessinNoir.x + dessinNoir.width > img.cols) dessinNoir.width = img.cols - dessinNoir.x;
    if (dessinNoir.y + dessinNoir.height > img.rows) dessinNoir.height = img.rows - dessinNoir.y; 

    // On dessine dans backproj le rectangle noir
    rectangle(camShiftData.backproj, dessinNoir, CV_RGB(0,0,0), CV_FILLED, 1, 0); // CV_FILLED permet de dessiner un rectangle plein.

    rectangle( img, camShiftData.trackWindow, CV_RGB(0,255,0), 1, 8, 0 );

    // On definie le nouveau rectangle de recherche a gauche du visage (la main droite)
    Rect main = Rect( 0, 0, camShiftData.trackWindow.x, img.rows );

    // On reappelle camShift après avoir noirci le visage, si l'espace a gauche du visage est suffisement grand pour contenir une main
    if (main.width > (dessinNoir.width / 2)){
	    trackBox = CamShift(camShiftData.backproj, main,
	    					TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));    
	    
	    // On definie un nouveau rectangle pour cacher le bas du bras
	    Rect noir = Rect(0, main.y + main.height / 2 + 70, camShiftData.trackWindow.x, img.rows - main.y);
    	if (noir.y + noir.height > img.rows) noir.height = img.rows - noir.y;

    	// dessin sur img et backproj de ce rectangle
    	rectangle( camShiftData.backproj, noir, CV_RGB(0,0,0), CV_FILLED, 1, 0);
    	rectangle( img, noir, CV_RGB(0,255,255), 1, 8, 0 );

	    trackBox = CamShift(camShiftData.backproj, main,
	    		TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

		// On redimentionne le rectangle contenant la main pour ne pas avoir le reste du bras.
	    main.y -= 10;
        if (main.y < 0) main.y = 0;
	    main.width += 40;
	    main.height = main.width;
	    

	    // On dessine le rectangle et l'ellipse de suivit sur l'image
	    ellipse( img, trackBox, Scalar(0,0,255), 3, CV_AA );	
	    rectangle( img, main, CV_RGB(0,0,255), 1, 8, 0 );    	
	}

    // Les deux affichages importants
	imshow("result", img);
	imshow("blackproj", camShiftData.backproj);

	return main;
}