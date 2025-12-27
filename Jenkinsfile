pipeline {
    agent {
            node {  label 'MLAgent' }
    }

    stages {

        stage('Checkout') {
            steps {
                // Descarga el código del repositorio automáticamente
                checkout scm
            }
        }

        stage('Verificar Archivos') {
            steps {
                echo "Trabajando en: ${WORKSPACE}"
                bat 'dir' // Esto confirmará que ves tus archivos .py y el requirements.txt
            }
        }

        stage('Entorno e Instala dependencias') {
            steps {
                bat '''
                @echo off
                if not exist venv ( python -m venv venv )
                call venv\\Scripts\\activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Paso 1: Database (ETL)') {
            steps {
                bat '''
                @echo off
                call venv\\Scripts\\activate
                python -m src.jobs.database
                '''
            }
        }

        stage('Paso 2: Entrenamiento (Train)') {
            steps {
                bat '''
                @echo off
                call venv\\Scripts\\activate
                python -m src.jobs.train
                '''
            }
        }

        stage('Paso 3: Predicción (Predict)') {
            steps {
                bat '''
                @echo off
                call venv\\Scripts\\activate
                python -m src.jobs.predict
                '''
            }
        }
    }
}