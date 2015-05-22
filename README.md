Dupanloup Remy

Système d'exploitation : Ubuntu 14.04 LTS 64bits
OpenCV : version 2.4.10

Rendu numero 3

I) Letter_Recog : 

	1) Nettoyage du code :

	Suppression de toutes les méthodes qui ne concerne pas la partie mlp, on garde cependant la méthode read_num_class_data() qui est appelée dans build_mlp_classifier().

	Modification du main pour qu'il appelle automatiquement la méthode mlp avec les bons fichiers. Un appel de script est même appelé avant pour mélanger le fichier contenant les données de la main pour augmenter la précision lors de la génération du réseau de neurones.

	2) Modification de l'algorithme :

	Dans la fonction build_mlp_classifier() on change le nombre de class_count a 26 ce qui correspond aux 26 lettres de l'alphabet.
	On change également dans l'appel de la fonction read_num_class_data() le 2eme paramètre a 256, car une lettre est représenter par 256 valeurs.


II) Facedetect :

	1) Nettoyage du code :

	Aucun nettoyage supplémentaire n'a été effectué par rapport au rendu numero 2.

	2) Modification de l'algorithme :

	Dans le main, lorsque l'on se trouve dans la partie CamShiftData, j'ai rajouté un evenement correspondant un appel du clavier. Si l'utilisateur appuye sur une lettre du clavier, l'image de la main est sauvegarder dans le fichier image.txt avec la lettre correspondante. Si l'utilisateur appuye sur espace, le programme affiche sur la console la lettre reconnue.


III) Utilisation du programme :

	Pour effectuer l'apprentissage, il faut lancer facedetect, reconnaitre suffisament de lettre, en appuyant sur les lettres correspondantes au clavier, pour remplir le fichier image.txt. Il faut ensuite générer le réseau de neurone. Pour ce faire on utilise letter_recog qui charge automatiquement image.txt, le trie dans image_trie.txt et effectue le calcul du reseau de neurone et le génère dans neurones.xml.

	Relancer ensuite facedetect qui chargera le nouveau réseau de neurones mis a jour, et lors de l'appuye sur la touche espace, le programme ecrira dans la console la lettre détecter.


IV) Infos supplémentaires :

	Les deux principales lettres détecter sont le H et le I, la manière dont elle sont faites avec la main sont celles présentent dans l'image jointe avec les fichiers de rendu.
	Le fichier image.txt a été conçus uniquement par moi, et je suis le seul a l'utiliser.
	Dans le fichier image.txt il y a 35 fois chacune des deux lettres, et les tests dans letter_recog sont tous les deux à 100%.

	En ajoutant 40 fois la lettre B dans le fichier image.txt, les tests de letter_recog passe a 100% pour le train et 94% pour le test.

	La reconnaissance reste cependant très satisfaisante. 
