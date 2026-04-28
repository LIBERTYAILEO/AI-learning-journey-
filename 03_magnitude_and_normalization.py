

"""
Phase 1: Linear Algebra - Deep Dive
Module 3: Magnitude, Normalization & Cosine Similarity

This module explores why magnitude matters and how normalization
keeps comparisons fair and meaningful.

Key insight: Two vectors pointing in the SAME DIRECTION should be
considered similar, even if one is longer than the other.
"""

import math


# ============================================
# 1. MAGNITUDE EXPLORATION
# ============================================

class MagnitudeExplorer:
    """
    Understand what magnitude means and why it matters.
    """
    
    @staticmethod
    def magnitude(vector):
        """
        Calculate magnitude (length) of a vector.
        
        Formula: ||v|| = √(v₁² + v₂² + ... + vₙ²)
        
        This is just the Pythagorean theorem in n-dimensions!
        """
        return math.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def demonstrate_magnitude():
        """Show what magnitude represents geometrically."""
        print("\n" + "="*70)
        print("MAGNITUDE: What does it mean?")
        print("="*70)
        
        # 2D example (easy to visualize)
        v1 = [3, 4]
        v2 = [6, 8]  # Same direction, but twice as long
        v3 = [1, 1]  # Different direction
        
        mag_v1 = MagnitudeExplorer.magnitude(v1)
        mag_v2 = MagnitudeExplorer.magnitude(v2)
        mag_v3 = MagnitudeExplorer.magnitude(v3)
        
        print(f"\nVector v1 = {v1}")
        print(f"  Magnitude: {mag_v1}")
        print(f"  Think of it as: How far from origin (0,0)?")
        print(f"  Calculation: √(3² + 4²) = √(9 + 16) = √25 = 5")
        
        print(f"\nVector v2 = {v2}")
        print(f"  Magnitude: {mag_v2}")
        print(f"  Same direction as v1, but TWICE as long!")
        print(f"  Calculation: √(6² + 8²) = √(36 + 64) = √100 = 10")
        
        print(f"\nVector v3 = {v3}")
        print(f"  Magnitude: {mag_v3:.3f}")
        print(f"  Different direction from v1 and v2")
        print(f"  Calculation: √(1² + 1²) = √2 ≈ 1.414")
        
        print(f"\n{'Key Insight:':<20} v1 and v2 point in SAME direction")
        print(f"{'':20} but v2 is longer")
        print(f"{'':20} They should still be considered similar!")


# ============================================
# 2. THE PROBLEM WITH RAW DOT PRODUCT
# ============================================

class DotProductProblem:
    """
    Show why raw dot product can be misleading.
    """
    
    @staticmethod
    def dot_product(v1, v2):
        """Calculate dot product."""
        return sum(a * b for a, b in zip(v1, v2))
    
    @staticmethod
    def magnitude(vector):
        """Calculate magnitude."""
        return math.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def demonstrate_problem():
        """Show the problem with scale."""
        print("\n" + "="*70)
        print("THE PROBLEM: Raw Dot Product is Scale-Sensitive")
        print("="*70)
        
        # User 1 and User 2 want similar movies
        user1_prefs = [1, 0, 1]    # Likes action (1), not romance (0), likes sci-fi (1)
        user2_prefs = [10, 0, 10]  # Same preferences but all values scaled by 10
        
        # Movie characteristics
        action_movie = [1, 0, 0]
        romance_movie = [0, 1, 0]
        
        print(f"\nUser 1 preferences: {user1_prefs} (action=1, romance=0, sci-fi=1)")
        print(f"User 2 preferences: {user2_prefs} (action=10, romance=0, sci-fi=10)")
        print(f"\nThey have the SAME TASTES, just different scales!")
        
        # Raw dot products
        dp1_action = DotProductProblem.dot_product(user1_prefs, action_movie)
        dp2_action = DotProductProblem.dot_product(user2_prefs, action_movie)
        
        print(f"\n--- Raw Dot Products (MISLEADING) ---")
        print(f"User 1 · Action Movie = {dp1_action}")
        print(f"User 2 · Action Movie = {dp2_action}")
        print(f"\nProblem: User 2 gets 10x higher score!")
        print(f"But they have IDENTICAL TASTES!")
        print(f"The difference is just SCALE, not actual preference difference.")
        
        # Magnitude
        mag_user1 = DotProductProblem.magnitude(user1_prefs)
        mag_user2 = DotProductProblem.magnitude(user2_prefs)
        
        print(f"\n--- Magnitude (Length) ---")
        print(f"Magnitude of User 1: {mag_user1:.3f}")
        print(f"Magnitude of User 2: {mag_user2:.3f}")
        print(f"\nUser 2's magnitude is 10x larger!")
        print(f"That's why the dot product is 10x larger too!")


# ============================================
# 3. SOLUTION: NORMALIZATION
# ============================================

class Normalization:
    """
    Normalize vectors to fix the scale problem.
    """
    
    @staticmethod
    def magnitude(vector):
        """Calculate magnitude."""
        return math.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def normalize(vector):
        """
        Normalize a vector to unit length (magnitude = 1).
        
        Formula: v_normalized = v / ||v||
        
        This removes the scale effect while keeping direction!
        """
        mag = Normalization.magnitude(vector)
        
        if mag == 0:
            return vector  # Can't normalize zero vector
        
        return [x / mag for x in vector]
    
    @staticmethod
    def dot_product(v1, v2):
        """Calculate dot product."""
        return sum(a * b for a, b in zip(v1, v2))
    
    @staticmethod
    def cosine_similarity(v1, v2):
        """
        Calculate cosine similarity (dot product of normalized vectors).
        
        Formula: cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
        
        Or equivalently: normalize both vectors, then dot product them.
        
        Range: -1 to 1
        - 1.0 = same direction
        - 0.0 = perpendicular
        - -1.0 = opposite direction
        """
        mag_v1 = Normalization.magnitude(v1)
        mag_v2 = Normalization.magnitude(v2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        dot_prod = Normalization.dot_product(v1, v2)
        return dot_prod / (mag_v1 * mag_v2)
    
    @staticmethod
    def demonstrate_normalization():
        """Show how normalization solves the scale problem."""
        print("\n" + "="*70)
        print("SOLUTION: Normalization & Cosine Similarity")
        print("="*70)
        
        user1_prefs = [1, 0, 1]
        user2_prefs = [10, 0, 10]
        action_movie = [1, 0, 0]
        
        print(f"\nUser 1 preferences: {user1_prefs}")
        print(f"User 2 preferences: {user2_prefs} (10x scaled)")
        print(f"Action movie: {action_movie}")
        
        # Normalize
        user1_norm = Normalization.normalize(user1_prefs)
        user2_norm = Normalization.normalize(user2_prefs)
        action_norm = Normalization.normalize(action_movie)
        
        print(f"\n--- After Normalization (unit length) ---")
        print(f"User 1 normalized: {[f'{x:.3f}' for x in user1_norm]}")
        print(f"User 2 normalized: {[f'{x:.3f}' for x in user2_norm]}")
        print(f"Action normalized: {[f'{x:.3f}' for x in action_norm]}")
        
        # Cosine similarities
        cos_sim_1 = Normalization.cosine_similarity(user1_prefs, action_movie)
        cos_sim_2 = Normalization.cosine_similarity(user2_prefs, action_movie)
        
        print(f"\n--- Cosine Similarity (CORRECT) ---")
        print(f"User 1 vs Action: {cos_sim_1:.3f}")
        print(f"User 2 vs Action: {cos_sim_2:.3f}")
        print(f"\n✓ NOW they're EQUAL!")
        print(f"✓ Scale doesn't matter anymore!")
        print(f"✓ Only DIRECTION matters!")


# ============================================
# 4. REAL WORLD: WORD EMBEDDINGS
# ============================================

class WordEmbeddingAnalysis:
    """
    Real example: How word embeddings use cosine similarity.
    """
    
    @staticmethod
    def magnitude(vector):
        """Calculate magnitude."""
        return math.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def cosine_similarity(v1, v2):
        """Calculate cosine similarity."""
        mag_v1 = WordEmbeddingAnalysis.magnitude(v1)
        mag_v2 = WordEmbeddingAnalysis.magnitude(v2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        dot_prod = sum(a * b for a, b in zip(v1, v2))
        return dot_prod / (mag_v1 * mag_v2)
    
    @staticmethod
    def demonstrate_word_embeddings():
        """Show real-world application in NLP."""
        print("\n" + "="*70)
        print("REAL WORLD: Word Embeddings & Semantic Similarity")
        print("="*70)
        
        # Simplified word embeddings (in reality, these are 300+ dimensions)
        embeddings = {
            'king': [0.2, 0.5, -0.3, 0.8, 0.1],
            'queen': [0.25, 0.48, -0.28, 0.75, 0.12],  # Similar to 'king'
            'prince': [0.22, 0.52, -0.25, 0.78, 0.08],  # Also similar
            'banana': [0.01, -0.9, 0.7, -0.5, 0.95],    # Very different
            'fruit': [0.05, -0.85, 0.75, -0.45, 0.92],   # Related to banana
        }
        
        print(f"\nWord embeddings (vectors representing word meanings):")
        for word, vec in embeddings.items():
            mag = WordEmbeddingAnalysis.magnitude(vec)
            print(f"  '{word}': magnitude = {mag:.3f}")
        
        print(f"\n--- Cosine Similarity Between Words ---")
        
        # Calculate similarities
        pairs = [
            ('king', 'queen'),
            ('king', 'prince'),
            ('king', 'banana'),
            ('banana', 'fruit'),
        ]
        
        for word1, word2 in pairs:
            sim = WordEmbeddingAnalysis.cosine_similarity(
                embeddings[word1], 
                embeddings[word2]
            )
            print(f"'{word1}' vs '{word2}': {sim:.3f} {'⭐ Similar!' if sim > 0.8 else '✗ Different'}")
        
        print(f"\n--- Why Cosine Similarity Works ---")
        print(f"Even though 'king' and 'queen' embeddings have")
        print(f"different magnitudes, cosine similarity finds they")
        print(f"point in NEARLY THE SAME DIRECTION = semantically related!")


# ============================================
# 5. MATHEMATICAL INTUITION
# ============================================

class MathematicalIntuition:
    """
    Build intuition about why this works.
    """
    
    @staticmethod
    def demonstrate_intuition():
        """Explain the math geometrically."""
        print("\n" + "="*70)
        print("MATHEMATICAL INTUITION: Why This Works")
        print("="*70)
        
        print("""
        Imagine vectors as ARROWS in space:
        
        RAW DOT PRODUCT:
        ┌─────────────────────────────────────────────────────┐
        │ Vector A (length 5) pointing NORTHEAST              │
        │ Vector B (length 10) pointing NORTHEAST (2x longer) │
        │                                                      │
        │ Dot Product:                                         │
        │ - Raw: A · B = 50 (depends on scale!)              │
        │ - Normalized: Â · B̂ = 1.0 (same direction!)       │
        └─────────────────────────────────────────────────────┘
        
        COSINE SIMILARITY:
        ┌─────────────────────────────────────────────────────┐
        │ Formula: cos(θ) = (A · B) / (||A|| × ||B||)        │
        │                                                      │
        │ What it measures:                                    │
        │ - The ANGLE between two vectors                     │
        │ - NOT affected by their lengths                     │
        │ - Only by their DIRECTION                           │
        │                                                      │
        │ Result:                                              │
        │ - cos(0°) = 1.0    (same direction)                 │
        │ - cos(90°) = 0.0   (perpendicular)                  │
        │ - cos(180°) = -1.0 (opposite direction)             │
        └─────────────────────────────────────────────────────┘
        
        WHY THIS MATTERS FOR AI:
        ┌─────────────────────────────────────────────────────┐
        │ In recommendation systems:                           │
        │ - User preferences are vectors (e.g., from clicks)  │
        │ - Movie features are vectors                        │
        │ - Cosine similarity finds the BEST MATCHES          │
        │ - Scale doesn't matter (10 clicks vs 100 clicks)    │
        │                                                      │
        │ In NLP (ChatGPT):                                   │
        │ - Words are embedded as vectors                     │
        │ - Cosine similarity finds synonym relationships     │
        │ - "king" - "man" + "woman" ≈ "queen"               │
        │   (famous vector math relationship!)                │
        │                                                      │
        │ In Computer Vision:                                 │
        │ - Images are feature vectors                        │
        │ - Cosine similarity finds similar images            │
        │ - Face recognition uses this!                       │
        └─────────────────────────────────────────────────────┘
        """)


# ============================================
# 6. INTERACTIVE VISUALIZATION
# ============================================

class InteractiveComparison:
    """
    Let user compare different similarity measures.
    """
    
    @staticmethod
    def magnitude(vector):
        """Calculate magnitude."""
        return math.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def dot_product(v1, v2):
        """Calculate dot product."""
        return sum(a * b for a, b in zip(v1, v2))
    
    @staticmethod
    def cosine_similarity(v1, v2):
        """Calculate cosine similarity."""
        mag_v1 = InteractiveComparison.magnitude(v1)
        mag_v2 = InteractiveComparison.magnitude(v2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        return InteractiveComparison.dot_product(v1, v2) / (mag_v1 * mag_v2)
    
    @staticmethod
    def euclidean_distance(v1, v2):
        """Calculate Euclidean distance (another similarity measure)."""
        return math.sqrt(sum((a - b)**2 for a, b in zip(v1, v2)))
    
    @staticmethod
    def compare_similarity_measures():
        """Compare different ways to measure similarity."""
        print("\n" + "="*70)
        print("COMPARISON: Different Similarity Measures")
        print("="*70)
        
        v1 = [1, 0, 0]
        v2 = [2, 0, 0]  # Same direction, twice as long
        v3 = [1, 1, 1]  # Different direction
        
        print(f"\nVector 1: {v1}")
        print(f"Vector 2: {v2} (same direction as v1, but 2x longer)")
        print(f"Vector 3: {v3} (different direction)")
        
        print(f"\n{'Measure':<25} {'v1 vs v2':<15} {'v1 vs v3':<15}")
        print("-" * 55)
        
        dp_12 = InteractiveComparison.dot_product(v1, v2)
        dp_13 = InteractiveComparison.dot_product(v1, v3)
        print(f"{'Raw Dot Product':<25} {dp_12:<15.3f} {dp_13:<15.3f}")
        print(f"{'':25} (high, scale dependent) (depends on scale)")
        
        cos_12 = InteractiveComparison.cosine_similarity(v1, v2)
        cos_13 = InteractiveComparison.cosine_similarity(v1, v3)
        print(f"\n{'Cosine Similarity':<25} {cos_12:<15.3f} {cos_13:<15.3f}")
        print(f"{'':25} (perfect match!)     (perpendicular)")
        
        dist_12 = InteractiveComparison.euclidean_distance(v1, v2)
        dist_13 = InteractiveComparison.euclidean_distance(v1, v3)
        print(f"\n{'Euclidean Distance':<25} {dist_12:<15.3f} {dist_13:<15.3f}")
        print(f"{'':25} (scale dependent)   (depends on scale)")
        
        print(f"\n{'Key Takeaway:':<25} Cosine similarity is scale-invariant!")
        print(f"{'':25} It only measures ANGLE, not LENGTH!")


# ============================================
# 7. RUN ALL DEMONSTRATIONS
# ============================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "DEEP DIVE: MAGNITUDE & NORMALIZATION" + " "*17 + "║")
    print("║" + " "*17 + "Why Direction Matters More Than Scale" + " "*13 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Run all demonstrations
    MagnitudeExplorer.demonstrate_magnitude()
    DotProductProblem.demonstrate_problem()
    Normalization.demonstrate_normalization()
    WordEmbeddingAnalysis.demonstrate_word_embeddings()
    MathematicalIntuition.demonstrate_intuition()
    InteractiveComparison.compare_similarity_measures()
    
    print("\n" + "="*70)
    print("SUMMARY: What You've Learned")
    print("="*70)
    print("""
    1. MAGNITUDE tells you the LENGTH of a vector
       - ||v|| = √(v₁² + v₂² + ... + vₙ²)
    
    2. RAW DOT PRODUCT is scale-dependent
       - Longer vectors give larger dot products
       - This can be misleading!
    
    3. NORMALIZATION removes scale
       - v_normalized = v / ||v||
       - Makes magnitude = 1 (unit vector)
    
    4. COSINE SIMILARITY is scale-invariant
       - cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)
       - Only measures DIRECTION
       - Range: -1 to 1
    
    5. APPLICATIONS IN AI:
       - Recommendations: Compare user & movie vectors
       - NLP: Find similar words/meanings
       - Computer Vision: Find similar images
       - All scale-invariant because we only care about direction!
    
    YOUR INSIGHT WAS SPOT ON:
    ✓ You understood that scaling shouldn't change similarity
    ✓ You realized we need to normalize to compare fairly
    ✓ You grasped the concept of "average over sample"
    
    This is exactly what cosine similarity does!
    """)