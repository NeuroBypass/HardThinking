"""
Adapter para modelos de Machine Learning usando TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pickle
import os
from typing import Dict, Any, Optional
from ...application.ports.secondary_ports import MLModelPort

class TensorFlowMLAdapter(MLModelPort):
    """Adapter para TensorFlow/Keras"""
    
    def __init__(self, config):
        self.config = config
    
    def create_model(self, architecture: str, hyperparameters: Dict[str, Any]) -> tf.keras.Model:
        """Cria modelo TensorFlow baseado na arquitetura"""
        
        if architecture == "CNN_1D":
            return self._create_cnn_1d_model(hyperparameters)
        elif architecture == "EEGNET":
            return self._create_eegnet_model(hyperparameters)
        else:
            raise ValueError(f"Arquitetura não suportada: {architecture}")
    
    def _create_cnn_1d_model(self, hyperparameters: Dict[str, Any]) -> tf.keras.Model:
        """Cria modelo CNN 1D"""
        
        # Parâmetros padrão
        input_shape = hyperparameters.get('input_shape', (250, 16))  # (window_size, channels)
        conv_filters = hyperparameters.get('conv_filters', [64, 64, 128])
        conv_kernels = hyperparameters.get('conv_kernels', [3, 3, 3])
        dense_units = hyperparameters.get('dense_units', 512)
        dropout_rates = hyperparameters.get('dropout_rates', [0.25, 0.25, 0.5])
        num_classes = hyperparameters.get('num_classes', 2)
        
        model = Sequential()
        
        # Primeira camada convolucional
        model.add(Conv1D(filters=conv_filters[0], 
                        kernel_size=conv_kernels[0], 
                        activation='relu',
                        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rates[0]))
        
        # Segunda camada convolucional
        model.add(Conv1D(filters=conv_filters[1], 
                        kernel_size=conv_kernels[1], 
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rates[1]))
        
        # Terceira camada convolucional
        model.add(Conv1D(filters=conv_filters[2], 
                        kernel_size=conv_kernels[2], 
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rates[1]))
        
        # Flatten e camadas densas
        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rates[2]))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compila modelo
        model.compile(
            optimizer=Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_eegnet_model(self, hyperparameters: Dict[str, Any]) -> tf.keras.Model:
        """Cria modelo EEGNet (implementação simplificada)"""
        
        input_shape = hyperparameters.get('input_shape', (250, 16))
        num_classes = hyperparameters.get('num_classes', 2)
        
        model = Sequential()
        
        # Bloco temporal
        model.add(Conv1D(filters=16, kernel_size=64, padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        
        # Bloco espacial
        model.add(Conv1D(filters=32, kernel_size=1))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.25))
        
        # Bloco separável
        model.add(Conv1D(filters=32, kernel_size=16, padding='same', groups=32))
        model.add(Conv1D(filters=32, kernel_size=1))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))
        model.add(MaxPooling1D(pool_size=8))
        model.add(Dropout(0.25))
        
        # Classificador
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer=Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Treina modelo TensorFlow"""
        
        # Converte labels para categorical
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=2)
        
        # Reshape X se necessário
        if len(X.shape) == 2:
            # X está como (samples, features), precisa reshapear para (samples, time_steps, channels)
            samples, features = X.shape
            channels = 16  # Assumindo 16 canais
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Treina modelo
        history = model.fit(
            X, y_categorical,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'history': history.history,
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
    
    def predict(self, model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        
        # Reshape X se necessário
        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)
        
        predictions = model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
        """Faz predições com probabilidades"""
        
        # Reshape X se necessário
        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)
        
        return model.predict(X, verbose=0)
    
    def evaluate_model(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Avalia modelo"""
        
        # Converte labels para categorical
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=2)
        
        # Reshape X se necessário
        if len(X.shape) == 2:
            samples, features = X.shape
            channels = 16
            time_steps = features // channels
            X = X.reshape(samples, time_steps, channels)
        
        loss, accuracy = model.evaluate(X, y_categorical, verbose=0)
        
        # Predições para métricas adicionais
        y_pred = self.predict(model, X)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
    
    def save_model(self, model: tf.keras.Model, file_path: str) -> bool:
        """Salva modelo em arquivo"""
        try:
            # Cria diretório se não existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Salva modelo
            model.save(file_path)
            return True
        except Exception as e:
            print(f"Erro ao salvar modelo: {str(e)}")
            return False
    
    def load_model(self, file_path: str) -> Optional[tf.keras.Model]:
        """Carrega modelo de arquivo"""
        try:
            if os.path.exists(file_path):
                return load_model(file_path)
            else:
                print(f"Arquivo não encontrado: {file_path}")
                return None
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return None
