def print_matrix(matrix):
    """
    Prints a 2D matrix in a readable format.
    
    Args:
        matrix (list of list of int/float): The matrix to be printed.
    """
    for row in matrix:
        print(" ".join(map(str, row)))

def TestAll():
    """
    Function to test various functionalities.
    """
    # Example test for print_matrix
    test_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print("Testing print_matrix with a 3x3 matrix:")
    print_matrix(test_matrix)
    print("Test completed.")

# Example usage
if __name__ == "__main__":
    TestAll()