import json

import pytest
from sklearn.ensemble import RandomForestRegressor

from app import Prediction, User, create_app, db
from model import (
    FEATURE_COLUMNS,
    generate_student_dataset,
    save_dataset,
    train_model,
    validate_student_features,
)


VALID_FEATURES = {
    "Attendance": "85",
    "StudyHours": "4",
    "InternetUsage": "3",
    "SleepHours": "7",
    "PreviousMarks": "80",
    "AssignmentScore": "82",
    "ClassParticipation": "8",
    "ParentalSupport": "1",
    "ExtraActivities": "1",
}


class StubModel:
    def predict(self, rows):
        return [81.25]


@pytest.fixture
def app(tmp_path):
    test_app = create_app(
        test_config={
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
            "SECRET_KEY": "test-secret",
        },
        model=StubModel(),
    )
    yield test_app


@pytest.fixture
def client(app):
    return app.test_client()


def register_and_login(client, email="student@example.com", username="student"):
    client.post(
        "/register",
        data={
            "username": username,
            "email": email,
            "password": "password123",
            "confirm": "password123",
        },
        follow_redirects=True,
    )
    return client.post(
        "/login",
        data={"email": email, "password": "password123"},
        follow_redirects=True,
    )


def test_validate_all_nine_features_required():
    values, error = validate_student_features({"Attendance": "80"})
    assert values is None
    assert "Missing required features" in error


def test_validate_feature_range():
    payload = dict(VALID_FEATURES)
    payload["Attendance"] = "150"
    values, error = validate_student_features(payload)
    assert values is None
    assert "Attendance" in error


def test_validate_binary_features():
    payload = dict(VALID_FEATURES)
    payload["ParentalSupport"] = "2"
    values, error = validate_student_features(payload)
    assert values is None
    assert "ParentalSupport" in error


def test_validate_accepts_complete_input():
    values, error = validate_student_features(VALID_FEATURES)
    assert error is None
    assert len(values) == 9
    assert values[0] == 85


def test_register_and_login(client):
    response = register_and_login(client)
    assert response.status_code == 200
    assert b"Dash" in response.data


def test_predict_rejects_incomplete_features(client):
    register_and_login(client)
    incomplete = dict(VALID_FEATURES)
    incomplete.pop("AssignmentScore")
    response = client.post("/predict_page", data=incomplete)
    assert response.status_code == 400
    assert b"Missing required features" in response.data


def test_prediction_is_saved_for_logged_in_user(client, app):
    register_and_login(client)
    response = client.post("/predict_page", data=VALID_FEATURES)
    assert response.status_code == 302
    assert "/result_page/" in response.headers["Location"]

    with app.app_context():
        user = User.query.filter_by(email="student@example.com").first()
        records = Prediction.query.filter_by(user_id=user.id).all()
        assert len(records) == 1
        assert records[0].prediction == 81.25
        stored = json.loads(records[0].feature_data)
        assert len(stored) == 9


def test_history_is_user_specific(client, app):
    register_and_login(client, email="one@example.com", username="one")
    client.post("/predict_page", data=VALID_FEATURES)
    client.get("/logout")

    register_and_login(client, email="two@example.com", username="two")
    history = client.get("/history")
    assert history.status_code == 200
    assert b"81.25" not in history.data
    assert b"No predictions yet" in history.data

    client.post("/predict_page", data=VALID_FEATURES)
    history = client.get("/history")
    assert b"81.25" in history.data

    with app.app_context():
        user_one = User.query.filter_by(email="one@example.com").first()
        user_two = User.query.filter_by(email="two@example.com").first()
        assert Prediction.query.filter_by(user_id=user_one.id).count() == 1
        assert Prediction.query.filter_by(user_id=user_two.id).count() == 1


def test_result_page_uses_prediction_id_and_blocks_other_users(client, app):
    register_and_login(client, email="owner@example.com", username="owner")
    predict = client.post("/predict_page", data=VALID_FEATURES)
    result_url = predict.headers["Location"]
    owned = client.get(result_url)
    assert owned.status_code == 200
    assert b"81.25" in owned.data

    with app.app_context():
        prediction_id = Prediction.query.first().id

    client.get("/logout")
    register_and_login(client, email="other@example.com", username="other")
    blocked = client.get(f"/result_page/{prediction_id}", follow_redirects=False)
    assert blocked.status_code == 302
    assert "/history" in blocked.headers["Location"]


def test_history_requires_login(client):
    response = client.get("/history", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_result_page_without_id_shows_latest_prediction(client):
    register_and_login(client)
    client.post("/predict_page", data=VALID_FEATURES)
    response = client.get("/result_page")
    assert response.status_code == 200
    assert b"81.25" in response.data


def test_dataset_has_nine_features_and_is_large():
    data = generate_student_dataset(n_samples=1000, seed=42)
    assert len(data) == 1000
    assert list(data.columns[:9]) == FEATURE_COLUMNS
    assert "FinalPerformance" in data.columns


def test_model_training_reports_mae_rmse_r2_and_cross_validation(tmp_path):
    csv_path = tmp_path / "student_data.csv"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "model_metrics.json"
    plot_path = tmp_path / "feature_importance.png"

    save_dataset(csv_path, n_samples=250, seed=42)
    model, metrics = train_model(
        csv_path=csv_path,
        model_path=model_path,
        metrics_path=metrics_path,
        importance_plot=plot_path,
        n_estimators=40,
        random_state=42,
    )

    assert isinstance(model, RandomForestRegressor)
    assert metrics["n_samples"] == 250
    assert metrics["test_mae"] >= 0
    assert metrics["test_rmse"] >= metrics["test_mae"]
    assert -1 <= metrics["test_r2"] <= 1
    assert metrics["cv_splits"] == 5
    assert "cv_r2_mean" in metrics
    assert metrics["test_r2"] > 0.6
    assert model_path.exists()
    assert json.loads(metrics_path.read_text())["test_rmse"] == metrics["test_rmse"]
