# Étape de build
FROM node:18-alpine AS build-stage

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de définition de dépendances
COPY package*.json ./

# Installer les dépendances en utilisant npm ci pour des builds plus rapides et déterministes
RUN npm ci --silent

# Copier tous les fichiers du projet
COPY . .

# Build de l'application
RUN npm run build

# Étape de production
FROM nginx:alpine

# Copier la configuration Nginx personnalisée
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Copier les fichiers buildés de l'étape précédente
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Configuration des ports et permissions
EXPOSE 80

# Commande par défaut
CMD ["nginx", "-g", "daemon off;"]