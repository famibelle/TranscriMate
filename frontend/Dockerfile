# Étape 1 : Utiliser une image Node.js pour la construction
FROM node:18 as build

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers package.json et package-lock.json
COPY package*.json ./

# Installer les dépendances
RUN npm install

# Copier tout le reste du projet
COPY . .

# Construire le projet pour la production
RUN npm run build

# Étape 2 : Utiliser une image nginx pour servir les fichiers statiques
FROM nginx:alpine

# Copier les fichiers de build générés par l'étape précédente
COPY --from=build /app/dist /usr/share/nginx/html

# Exposer le port utilisé par Nginx
EXPOSE 8080

# Démarrer Nginx
CMD ["nginx", "-g", "daemon off;"]
