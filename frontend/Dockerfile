# Étape 1 : Construire l'application frontend
FROM node:18.3.0 AS builder

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires pour la construction
COPY package*.json ./

# Installer les dépendances
RUN npm install

# Copier le reste du code source
COPY . .

# Construire l'application
RUN npm run build

# Étape 2 : Exécuter l'application dans un conteneur
# Vous pouvez choisir un serveur comme nginx ou continuer avec Node.js pour le développement.

# Option 1 : Utiliser un serveur Node.js
FROM node:18.3.0

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers générés par l'étape de construction
COPY --from=builder /app .

# Exposer le port pour servir l'application
EXPOSE 8080

# Commande pour démarrer l'application
CMD ["npm", "run", "serve", "--", "--host", "0.0.0.0", "--port", "8080", "--disable-host-check"]

# Option 2 : Utiliser Nginx pour servir les fichiers statiques (recommandé pour la production)
# Uncomment the lines below if you want to use Nginx
# FROM nginx:alpine
# COPY --from=builder /app/dist /usr/share/nginx/html
# EXPOSE 80
# CMD ["nginx", "-g", "daemon off;"]
