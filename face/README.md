# Face project — déploiement sur GitHub

1) Installer les dépendances (recommandé dans un venv) :
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Préparer le dépôt local et pousser sur GitHub :
   ./deploy.sh git@github.com:VOTRE_UTILISATEUR/VOTRE_REPO.git main

   Ou manuellement :
   git init
   git checkout -b main
   git add .
   git commit -m "Initial commit"
   git remote add origin git@github.com:VOTRE_UTILISATEUR/VOTRE_REPO.git
   git push -u origin main

Remarques :
- Assurez-vous d'avoir accès à GitHub (clé SSH ou token si HTTPS).
- Le script deploy.sh nécessite que vous fournissiez l'URL du remote.
- Je ne peux pas pousser pour vous ; exécutez le script depuis /Users/User/Desktop/face sur votre machine.
