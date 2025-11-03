import pandas as pd
import numpy as np
import traceback

try:
    from Library.Tools.Train_Split import split

    (X_train, X_test), (y_train, y_test) = split()
    input()

except Exception as e:
    print(f"Произошла ошибка: {e}")
    print("\nПодробный traceback:")
    traceback.print_exc()
    input()