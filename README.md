# AI Learning Journey

   ## Learning Roadmap

   ### Phase 1: Linear Algebra
   - Basics of Matrices
   - Vector Spaces
   - Matrix Decompositions

   ### Phase 2: Calculus
   - Limits and Continuity
   - Derivatives and Integrals
   - Multivariable Calculus

   ### Phase 3: Probability & Statistics
   - Descriptive Statistics
   - Probability Distributions
   - Statistical Inference

   ### Phase 4: Neural Networks
   - Introduction to Neural Networks
   - Training Neural Networks
   - Advanced Topics in Neural Networks





## Phase 1: Linear Algebra - Foundation of AI

Linear algebra is the language of AI. Everything from image processing to neural networks relies on it. 


### What we'll cover:

1. ### Vectors & Matrices - The building blocks
2. ### Matrix Operations - Addition, multiplication, transposition
3. ### Dot Products & Vector Spaces - Understanding relationships
4. ### Matrix Decompositions - Breaking down complex data

### Let's Start: Vectors & Matrices 

### Concept 1: What is a Vector?

A vector is an ordered list of numbers. Think of it as a point in space or a direction.

Example:

v = [1, 2, 3]  # A 3D vector


### Concept 2: What is a Matrix?

A matrix is a 2D array of numbers arranged in rows and columns.

Example:

A = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
# This is a 3x3 matrix (3 rows, 3 columns)

### Now Let's Code It Together

I'll create a Python file that implements these concepts from scratch, and then explain the math:

### Link 🔗 to Code Below : 
https://github.com/LIBERTYAILEO/AI-learning-journey-/blob/main/01_vectors_and_matrices.py

```
"""
Phase 1: Linear Algebra Basics
Module 1: Vectors and Matrices

This module explores the fundamental concepts of vectors and matrices,
which form the foundation of all AI/ML operations.
"""

import math

# ============================================
# 1. VECTOR OPERATIONS
# ============================================

class Vector:
    """A simple Vector class to understand vector operations."""
    
    def __init__(self, values):
        """
        Initialize a vector with a list of values.
        
        Args:
            values (list): List of numbers representing the vector
            
        Example:
            v = Vector([1, 2, 3])
        """
        self.values = values
        self.dimension = len(values)
    
    def __repr__(self):
        return f"Vector({self.values})"
    
    def magnitude(self):
        """
        Calculate the magnitude (length) of the vector.
        
        Formula: ||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
        
        Returns:
            float: The magnitude of the vector
            
        Example:
            v = Vector([3, 4])
            v.magnitude()  # Returns 5 (because sqrt(3^2 + 4^2) = 5)
        """
        sum_of_squares = sum(x**2 for x in self.values)
        return math.sqrt(sum_of_squares)
    
    def dot_product(self, other):
        """
        Calculate dot product with another vector.
        
        Formula: v · w = v1*w1 + v2*w2 + ... + vn*wn
        
        Args:
            other (Vector): Another vector
            
        Returns:
            float: The dot product
            
        Example:
            v = Vector([1, 2, 3])
            w = Vector([4, 5, 6])
            v.dot_product(w)  # Returns 32 (1*4 + 2*5 + 3*6)
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        return sum(a * b for a, b in zip(self.values, other.values))
    
    def add(self, other):
        """
        Add two vectors element-wise.
        
        Formula: v + w = [v1+w1, v2+w2, ..., vn+wn]
        
        Args:
            other (Vector): Another vector
            
        Returns:
            Vector: The sum vector
        """
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        result = [a + b for a, b in zip(self.values, other.values)]
        return Vector(result)
    
    def scalar_multiply(self, scalar):
        """
        Multiply vector by a scalar (single number).
        
        Formula: c * v = [c*v1, c*v2, ..., c*vn]
        
        Args:
            scalar (float): A number to multiply by
            
        Returns:
            Vector: The scaled vector
        """
        result = [scalar * x for x in self.values]
        return Vector(result)


# ============================================
# 2. MATRIX OPERATIONS
# ============================================

class Matrix:
    """A simple Matrix class to understand matrix operations."""
    
    def __init__(self, data):
        """
        Initialize a matrix with 2D list.
        
        Args:
            data (list of lists): 2D list representing the matrix
            
        Example:
            m = Matrix([[1, 2], [3, 4]])
        """
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    def __repr__(self):
        return f"Matrix({self.rows}x{self.cols})"
    
    def transpose(self):
        """
        Transpose the matrix (swap rows and columns).
        
        Example:
            A = [[1, 2],      A^T = [[1, 3],
                 [3, 4]]             [2, 4]]
        
        Returns:
            Matrix: The transposed matrix
        """
        transposed = [[self.data[j][i] for j in range(self.rows)] 
                      for i in range(self.cols)]
        return Matrix(transposed)
    
    def matrix_multiply(self, other):
        """
        Multiply two matrices.
        
        Formula: (A * B)[i,j] = sum of A[i,k] * B[k,j] for all k
        
        Args:
            other (Matrix): Another matrix
            
        Returns:
            Matrix: The product matrix
            
        Note: Number of columns in self must equal number of rows in other
        """
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply {self.rows}x{self.cols} matrix "
                f"with {other.rows}x{other.cols} matrix"
            )
        
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                # Calculate dot product of row i and column j
                element = sum(
                    self.data[i][k] * other.data[k][j] 
                    for k in range(self.cols)
                )
                row.append(element)
            result.append(row)
        
        return Matrix(result)
    
    def print_matrix(self):
        """Pretty print the matrix."""
        for row in self.data:
            print([f"{x:6.2f}" for x in row])


# ============================================
# 3. EXAMPLE USAGE & DEMONSTRATIONS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: LINEAR ALGEBRA BASICS")
    print("=" * 60)
    
    # --- Vector Operations ---
    print("\n1. VECTOR OPERATIONS")
    print("-" * 60)
    
    v1 = Vector([3, 4])
    print(f"\nVector v1: {v1}")
    print(f"Magnitude of v1: {v1.magnitude()}")
    print(f"Why? sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5")
    
    v2 = Vector([1, 2])
    print(f"\nVector v2: {v2}")
    
    dot_prod = v1.dot_product(v2)
    print(f"Dot product (v1 · v2): {dot_prod}")
    print(f"Why? (3*1) + (4*2) = 3 + 8 = 11")
    
    v3 = v1.add(v2)
    print(f"\nVector addition (v1 + v2): {v3}")
    print(f"Why? [3+1, 4+2] = [4, 6]")
    
    v4 = v1.scalar_multiply(2)
    print(f"Scalar multiplication (2 * v1): {v4}")
    print(f"Why? [2*3, 2*4] = [6, 8]")
    
    # --- Matrix Operations ---
    print("\n\n2. MATRIX OPERATIONS")
    print("-" * 60)
    
    A = Matrix([[1, 2, 3],
                [4, 5, 6]])
    print(f"\nMatrix A ({A.rows}x{A.cols}):")
    A.print_matrix()
    
    print(f"\nTranspose of A (A^T) ({A.transpose().rows}x{A.transpose().cols}):")
    A.transpose().print_matrix()
    
    B = Matrix([[1, 2],
                [3, 4],
                [5, 6]])
    print(f"\nMatrix B ({B.rows}x{B.cols}):")
    B.print_matrix()
    
    C = A.matrix_multiply(B)
    print(f"\nMatrix multiplication (A * B) ({C.rows}x{C.cols}):")
    C.print_matrix()
    print("Why? First element = (1*1 + 2*3 + 3*5) = 22")
    
    print("\n" + "=" * 60)
```


### Why Dot Products Matter in AI


### The Big Picture:

Dot products are the fundamental operation that makes neural networks work. Every prediction, every classification, every decision an AI model makes comes down to dot products.


### Let me show you 3 concrete examples:



### Example 1: Image Classification (What is this image?)

Imagine you want to classify a handwritten digit (0-9).

```



Image pixel values: 

[0.5, 0.9, 0.2, 0.1, 0.8, ...]  (784 values for 28x28 image)

Neural Network weights: 

[0.3, -0.5, 0.7, 0.2, 0.1, ...]


Output = Dot Product = (0.5×0.3) + (0.9×-0.5) + (0.2×0.7) + ... 
       = 0.15 - 0.45 + 0.14 + ...
       = Some number that says "This looks like a 3"
```

The neural network learned weights that, when dot-producted with image pixels, give predictions!


### Example 2: Similarity Between Words (NLP)


In language models like ChatGPT:


```

Word "King":    [0.2, 0.5, -0.3, 0.1, ...]
Word "Queen":   [0.25, 0.48, -0.28, 0.12, ...]

Dot Product = High score → Words are similar!

Word "King":    [0.2, 0.5, -0.3, 0.1, ...]
Word "Banana":  [0.01, -0.9, 0.7, -0.5, ...]

Dot Product = Low score → Words are different
```


Dot products measure how related two concepts are!


### Example 3: Recommendation Systems (Netflix, YouTube)


```

User preferences: [likes_action, likes_romance, likes_horror, ...]
                = [0.9, 0.1, 0.2]

Movie characteristics: [is_action, is_romance, is_horror, ...]
                     = [0.8, 0.1, 0.1]

Dot Product = (0.9×0.8) + (0.1×0.1) + (0.2×0.1) = 0.73

High score → Recommend this movie!

```

### Now Let's Code This: Build a Simple Neural Network Layer

Let me create a file that shows dot products in action:

https://github.com/LIBERTYAILEO/AI-learning-journey-/blob/main/02_dot_product_in_neural_networks.py



        
