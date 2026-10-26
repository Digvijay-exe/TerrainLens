pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        DOCKERHUB_REPO = 'REPLACE_WITH_YOUR_DOCKERHUB_USERNAME/terrainlens'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t "$DOCKERHUB_REPO:$BUILD_NUMBER" .'
            }
        }

        stage('Run Automated Tests') {
            steps {
                sh 'docker run --rm "$DOCKERHUB_REPO:$BUILD_NUMBER" python -m pytest -q'
            }
        }

        stage('Push Image to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DH_USER',
                    passwordVariable: 'DH_TOKEN'
                )]) {
                    sh '''
                        echo "$DH_TOKEN" | docker login -u "$DH_USER" --password-stdin
                        docker push "$DOCKERHUB_REPO:$BUILD_NUMBER"
                        docker tag "$DOCKERHUB_REPO:$BUILD_NUMBER" "$DOCKERHUB_REPO:latest"
                        docker push "$DOCKERHUB_REPO:latest"
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh '''
                    kubectl config current-context
                    sed "s|IMAGE_PLACEHOLDER|$DOCKERHUB_REPO:$BUILD_NUMBER|g" \
                        k8s/deployment.yaml | kubectl apply -f -
                    kubectl apply -f k8s/service.yaml
                    kubectl rollout status deployment/terrainlens --timeout=180s
                    kubectl get deployments,pods,services
                '''
            }
        }
    }

    post {
        success {
            echo 'TerrainLens CI/CD pipeline completed successfully.'
        }
        failure {
            echo 'Pipeline failed. Inspect the failed stage in Jenkins Console Output.'
        }
    }
}
