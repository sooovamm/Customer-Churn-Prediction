
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_customer_data_schema():
    """CustomerData must accept valid numeric inputs."""
    from pydantic import ValidationError

  
    import sys, types

  
    sys.modules.setdefault("mlflow", MagicMock())
    sys.modules.setdefault("mlflow.pyfunc", MagicMock())
    sys.modules.setdefault("mysql", types.ModuleType("mysql"))
    sys.modules.setdefault("mysql.connector", MagicMock())

   
    from main import CustomerData

    data = CustomerData(
        gender=1, SeniorCitizen=0, Partner=1, Dependents=0,
        tenure=12.0, PhoneService=1, MultipleLines=0,
        InternetService=1, OnlineSecurity=0, OnlineBackup=1,
        DeviceProtection=0, TechSupport=0, StreamingTV=1,
        StreamingMovies=0, Contract=0, PaperlessBilling=1,
        PaymentMethod=2, MonthlyCharges=65.5, TotalCharges=786.0,
    )
    assert data.tenure == 12.0
    assert data.gender == 1


def test_customer_data_rejects_missing_field():
    """CustomerData must reject incomplete input."""
    import sys, types
    sys.modules.setdefault("mlflow", MagicMock())
    sys.modules.setdefault("mlflow.pyfunc", MagicMock())
    sys.modules.setdefault("mysql", types.ModuleType("mysql"))
    sys.modules.setdefault("mysql.connector", MagicMock())

    from pydantic import ValidationError
    from main import CustomerData

    with pytest.raises(ValidationError):
        CustomerData(gender=1)  


def test_feature_array_shape():
    """The feature array passed to the model must have 19 features."""
    features = [1, 0, 1, 0, 12.0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 2, 65.5, 786.0]
    X = np.array([features])
    assert X.shape == (1, 19), f"Expected (1, 19), got {X.shape}"