### Semantic Search

#### Contexte Stratégique

Les utilisateurs de MyGuichet.lu, la plateforme luxembourgeoise d'information administrative, rencontrent des difficultés pour trouver des informations précises rapidement et efficacement. La plateforme héberge plus de **10 992 pages** en trois langues principales, ce qui rend la recherche de réponses aux questions spécifiques complexe et laborieuse. La diversité linguistique et la spécificité des demandes administratives augmentent les défis d'accessibilité de l'information.

---

#### Objectif : Améliorer lExpérience Utilisateur par la Recherche Sémantique

Lobjectif de ce projet est dimplémenter une solution de recherche sémantique avancée pour répondre aux besoins d'information des utilisateurs de manière intuitive et précise. Les utilisateurs souhaitent pouvoir formuler leurs questions dans un langage naturel, sans se restreindre aux mots-clés exacts, et recevoir des réponses contextualisées qui tiennent compte de leur langue de préférence (luxembourgeois, français, portugais).

#### Enjeux : Limites des Fonctions de Recherche Actuelles

Les fonctionnalités de recherche actuelles ne capturent pas le sens véritable des requêtes des utilisateurs. Par exemple, une question telle que « Quels sont les règles de conduite pour les pharmaciens étrangers ? » peut ne produire aucun résultat sur Guichet.lu, malgré la présence d'informations pertinentes sur la plateforme. Les principaux défis sont :
- **Absence de recherche multilingue et sensible au contexte** : Les requêtes en langues mixtes ne sont pas bien interprétées.
- **Absence de compréhension sémantique** : Les recherches reposent sur des correspondances de mots-clés, ce qui limite la pertinence des résultats.
- **Expérience utilisateur insuffisante** : Lincapacité à fournir des résultats précis et rapides dégrade lexpérience des utilisateurs.

---

#### Solution : Mise en Place dune Recherche Sémantique Basée sur des Modèles d'IA

Pour résoudre ces problèmes, AKABI propose une solution de recherche sémantique intégrant des modèles de traitement du langage naturel avancés. Cette solution combine recherche vectorielle (recherche par embeddings) et algorithmes de correspondance BM25 pour offrir une expérience de recherche hybride. Les caractéristiques de cette solution incluent :
- **Recherche en langage naturel** : Permet aux utilisateurs de poser des questions dans leurs propres mots, améliorant ainsi l'accessibilité.
- **Gestion multilingue des requêtes** : La solution prend en charge plusieurs langues, notamment le luxembourgeois, le français et le portugais, pour offrir des réponses précises indépendamment de la langue dentrée.
- **Déploiement rapide et évolutif** : Grâce à Docker et CI/CD, la solution peut être déployée en moins de 5 minutes, garantissant une mise à jour continue et fluide.

#### Technologie et Infrastructure : Une Stack Technique Complète

AKABI met en place une infrastructure technique robuste pour supporter cette recherche sémantique. Les technologies clés incluent :
- **Modèles de recherche sémantique** : Intégration de modèles dembedding et de recherche hybride, combinant la recherche vectorielle avec BM25 pour des résultats optimaux.
- **Stack technique** : Utilisation de technologies avancées telles que **ChromaDB**, **Pinecone** pour le stockage d'embeddings, **GPT** et dautres LLMs (Large Language Models) pour la compréhension du langage naturel, ainsi que **Django**, **LangChain**, et **LlamaIndex** pour orchestrer larchitecture de l'application.
- **Interfaces utilisateur** : Développement dinterfaces conviviales via **Flutter** et **Streamlit** pour une expérience utilisateur fluide et intuitive.

---

#### Résultats Attendus : Amélioration de l'Accès à l'Information et de la Précision

La mise en uvre de cette solution sémantique vise des gains importants pour MyGuichet.lu et ses utilisateurs :
- **Réponses plus rapides et précises** : Grâce à l'intégration de la recherche sémantique, les utilisateurs recevront des réponses plus pertinentes et contextualisées.
- **Amélioration de lexpérience utilisateur** : Une meilleure compréhension des requêtes améliore la satisfaction des utilisateurs en leur permettant daccéder plus facilement aux informations.
- **Accès accru aux informations administratives** : Les utilisateurs pourront naviguer efficacement dans un vaste volume dinformations administratives, ce qui facilite la conformité et laccès aux services publics.

---

#### Ressources Humaines et Organisation

Pour garantir le succès de la mise en uvre, une équipe d'experts est mobilisée :
- **1 Data Scientist Senior** (6 mois)
- **1 Développeur Expert** (6 mois)
- **1 Data Engineer Expert** (6 mois)

---

### Conclusion

Le projet "Semantic Search" vise à transformer lexpérience de recherche des utilisateurs de MyGuichet.lu en leur offrant une solution moderne et performante. En intégrant une recherche contextuelle et multilingue, le gouvernement luxembourgeois s'assure de répondre aux besoins divers de ses citoyens, tout en renforçant lefficacité des services publics. Cette solution, en simplifiant laccès à des informations essentielles, contribue à renforcer la transparence et la réactivité de ladministration.