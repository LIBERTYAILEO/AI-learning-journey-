"""
Phase 1: Linear Algebra - Application
Module 2: Dot Products in Neural Networks

This module demonstrates how dot products are the core operation
in neural networks and AI systems.
"""

import math


# ============================================
# 1. SIMPLE NEURON (Using Dot Product)
# ============================================

class Neuron:
    """
    A single neuron that mimics how biological neurons work.
    
    In a real neural network:
    - Input: features/data
    - Weights: what the neuron learned
    - Dot Product: combines inputs with learned patterns
    - Activation: makes output non-linear
    """
    
    def __init__(self, weights, bias=0):
        """
        Initialize a neuron with weights and bias.
        
        Args:
            weights (list): Learned patterns the neuron recognizes
            bias (float): Threshold value
            
        Example:
            neuron = Neuron([0.5, -0.3, 0.8])
        """
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
        """
        Compute the neuron's output.
        
        Formula:
        output = dot_product(weights, inputs) + bias
        
        This is the dot product! It's how the neuron processes information.
        
        Args:
            inputs (list): Input features
            
        Returns:
            float: The neuron's activation (output)
        """
        if len(self.weights) != len(inputs):
            raise ValueError("Number of weights must match number of inputs")
        
        # THE MAGIC: Dot product of weights and inputs
        dot_product = sum(w * x for w, x in zip(self.weights, inputs))
        
        # Add bias and apply activation function
        output = dot_product + self.bias
        return output
    
    def forward_with_activation(self, inputs, activation='relu'):
        """
        Compute output with activation function (makes it non-linear).
        
        Args:
            inputs (list): Input features
            activation (str): Type of activation ('relu', 'sigmoid')
            
        Returns:
            float: The activated output
        """
        output = self.forward(inputs)
        
        if activation == 'relu':
            # ReLU: max(0, x) - zeros out negative values
            return max(0, output)
        elif activation == 'sigmoid':
            # Sigmoid: squashes output between 0 and 1
            return 1 / (1 + math.exp(-output))
        else:
            return output


# ============================================
# 2. SIMPLE IMAGE CLASSIFICATION
# ============================================

class SimpleImageClassifier:
    """
    A toy classifier for handwritten digits.
    Uses dot products to classify images.
    """
    
    def __init__(self):
        """Initialize with pre-trained weights (simulated)."""
        # Imagine we trained a neural network and got these weights
        # These weights represent what features matter for digit "3"
        self.neuron_for_3 = Neuron(
            weights=[0.1, 0.2, 0.15, 0.05, 0.3,  # First 5 pixel weights
                     0.25, 0.1, 0.2, 0.05, 0.15],  # More pixel weights
            bias=0.1
        )
    
    def classify_digit(self, image_pixels):
        """
        Classify if an image is a "3" or not.
        
        Args:
            image_pixels (list): Pixel values from the image
            
        Returns:
            dict: Contains score and prediction
        """
        # Get the score using dot product
        score = self.neuron_for_3.forward_with_activation(
            image_pixels, activation='sigmoid'
        )
        
        return {
            'score': score,
            'prediction': 'Likely a 3!' if score > 0.5 else 'Not a 3',
            'confidence': score
        }


# ============================================
# 3. WORD SIMILARITY (NLP Example)
# ============================================

class WordEmbedding:
    """
    Represents words as vectors (embeddings).
    Dot product measures similarity between words.
    """
    
    def __init__(self, embeddings):
        """
        Args:
            embeddings (dict): Maps word to vector
            
        Example:
            embeddings = {
                'king': [0.2, 0.5, -0.3],
                'queen': [0.25, 0.48, -0.28]
            }
        """
        self.embeddings = embeddings
    
    def similarity(self, word1, word2):
        """
        Compute similarity between two words.
        
        Higher dot product = more similar
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            float: Similarity score
        """
        if word1 not in self.embeddings or word2 not in self.embeddings:
            return None
        
        v1 = self.embeddings[word1]
        v2 = self.embeddings[word2]
        
        # Dot product of word vectors
        dot_prod = sum(a * b for a, b in zip(v1, v2))
        
        # Normalize by magnitudes (cosine similarity)
        mag1 = math.sqrt(sum(x**2 for x in v1))
        mag2 = math.sqrt(sum(x**2 for x in v2))
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_prod / (mag1 * mag2)


# ============================================
# 4. RECOMMENDATION SYSTEM
# ============================================

class RecommendationSystem:
    """
    A simple recommender using dot products.
    """
    
    def __init__(self, user_preferences, movie_features):
        """
        Args:
            user_preferences (list): User's preference vector
            movie_features (dict): Maps movie name to feature vector
        """
        self.user_preferences = user_preferences
        self.movie_features = movie_features
    
    def recommend_score(self, movie_name):
        """
        Calculate how much user might like a movie.
        
        Uses dot product: user_preferences · movie_features
        
        Args:
            movie_name (str): Name of the movie
            
        Returns:
            float: Recommendation score
        """
        if movie_name not in self.movie_features:
            return None
        
        features = self.movie_features[movie_name]
        
        # Dot product = recommendation score
        score = sum(u * f for u, f in zip(self.user_preferences, features))
        
        return score


# ============================================
# 5. DEMONSTRATIONS
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("WHY DOT PRODUCTS MATTER IN AI")
    print("=" * 70)
    
    # --- Example 1: Image Classification ---
    print("\n1. IMAGE CLASSIFICATION (Handwritten Digit Recognition)")
    print("-" * 70)
    
    classifier = SimpleImageClassifier()
    
    # Simulated image pixels for a digit that looks like "3"
    image_like_3 = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.1, 0.2, 0.05, 0.15]
    image_like_7 = [0.9, 0.1, 0.05, 0.02, 0.1, 0.05, 0.08, 0.9, 0.2, 0.1]
    
    result1 = classifier.classify_digit(image_like_3)
    print(f"\nImage that looks like '3':")
    print(f"  Score: {result1['score']:.3f}")
    print(f"  Prediction: {result1['prediction']}")
    print(f"  Why? Dot product of (learned weights) · (image pixels) = high score")
    
    result2 = classifier.classify_digit(image_like_7)
    print(f"\nImage that looks like '7':")
    print(f"  Score: {result2['score']:.3f}")
    print(f"  Prediction: {result2['prediction']}")
    print(f"  Why? Dot product doesn't match the learned '3' pattern")
    
    # --- Example 2: Word Similarity ---
    print("\n\n2. WORD SIMILARITY (Natural Language Processing)")
    print("-" * 70)
    
    embeddings = {
        'king': [0.2, 0.5, -0.3, 0.1],
        'queen': [0.25, 0.48, -0.28, 0.12],
        'man': [0.15, 0.45, -0.35, 0.05],
        'banana': [0.01, -0.9, 0.7, -0.5]
    }
    
    word_sim = WordEmbedding(embeddings)
    
    sim_king_queen = word_sim.similarity('king', 'queen')
    sim_king_banana = word_sim.similarity('king', 'banana')
    
    print(f"\nSimilarity between 'king' and 'queen': {sim_king_queen:.3f}")
    print(f"  ✓ High similarity! (related concepts)")
    print(f"  ✓ Measured using dot product of their embeddings")
    
    print(f"\nSimilarity between 'king' and 'banana': {sim_king_banana:.3f}")
    print(f"  ✗ Low similarity! (unrelated concepts)")
    
    # --- Example 3: Recommendation System ---
    print("\n\n3. RECOMMENDATION SYSTEM (Netflix, YouTube, Spotify)")
    print("-" * 70)
    
    # User likes: action, romance, sci-fi (with intensity 0-1)
    user_prefs = [0.9, 0.2, 0.8]  # Loves action, doesn't like romance much, likes sci-fi
    
    movies = {
        'Avatar': [0.8, 0.1, 0.9],      # Action, romance, sci-fi
        'Titanic': [0.2, 0.9, 0.1],     # Romance heavy
        'The Matrix': [0.7, 0.1, 0.95], # Action and sci-fi
        'Toy Story': [0.1, 0.2, 0.05]   # None of user's preferences
    }
    
    recommender = RecommendationSystem(user_prefs, movies)
    
    print(f"\nUser preferences: action=0.9, romance=0.2, sci-fi=0.8")
    print(f"\nRecommendation scores (using dot product):")
    
    for movie, score in sorted(
        [(m, recommender.recommend_score(m)) for m in movies.keys()],
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {movie:20} Score: {score:.2f} {'⭐ Recommended!' if score > 1.0 else ''}")
    
    print(f"\nWhy? Each movie's features are dot-producted with user preferences.")
    print(f"Higher dot product = better match = higher recommendation!")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
    Every AI system uses dot products because:
    
    1. Images → Dot product → Classification
    2. Words → Dot product → Similarity & Understanding
    3. Users & Movies → Dot product → Recommendations
    4. Sensor data → Dot product → Predictions
    5. Audio → Dot product → Speech Recognition
    
    Dot product is the UNIVERSAL OPERATION that makes AI work!
    """)