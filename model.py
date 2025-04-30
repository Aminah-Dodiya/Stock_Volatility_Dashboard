import os
from glob import glob
import pandas as pd
import joblib
from arch import arch_model
from data import AlphaVantageAPI, SQLRepository
from config import settings
import numpy as np
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

class GarchModel:
    """Class to manage GARCH model training and forecasting."""

    def __init__(self, ticker: str, repo: SQLRepository, use_new_data: bool, api_class: AlphaVantageAPI = None):
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.api_class = api_class if api_class else AlphaVantageAPI()
        self.model_directory = settings.model_directory

    def wrangle_data(self, n_observations: int) -> None:
        if self.use_new_data:
            new_data = self.api_class.get_daily(ticker=self.ticker)
            self.repo.insert_table(self.ticker, new_data, if_exists="replace")
        df = self.repo.read_table(self.ticker, limit=n_observations + 1)
        df.sort_index(ascending=True, inplace=True)
        df["return"] = df["close"].pct_change() * 100
        self.data = df["return"].dropna()

    def train(self, p: int, q: int) -> None:
        if not hasattr(self, "data"):
            raise Exception("No data found. Run 'wrangle_data()' first.")
        self.model = arch_model(self.data, p=p, q=q, rescale=False).fit(disp=0)
        self.aic = self.model.aic
        self.bic = self.model.bic
        self.resid = self.model.resid
        lb_test = acorr_ljungbox(self.resid, lags=[10], return_df=True)
        self.lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        self.train_time = pd.Timestamp.now()
        self.n_obs = len(self.data)

    def __clean_prediction(self, prediction: pd.DataFrame) -> dict:
        start = prediction.index[0] + pd.DateOffset(days=1)
        dates = pd.bdate_range(start=start, periods=prediction.shape[1])
        mean = pd.Series(prediction.values.flatten()**0.5, index=dates)
        stderr = mean / np.sqrt(len(mean))
        lower = mean - 1.96 * stderr
        upper = mean + 1.96 * stderr
        return {
            "mean": {str(k): float(v) for k, v in mean.round(4).items()},
            "lower": {str(k): float(v) for k, v in lower.round(4).items()},
            "upper": {str(k): float(v) for k, v in upper.round(4).items()}
        }

    def predict_volatility(self, horizon: int) -> dict:
        prediction = self.model.forecast(horizon=horizon, reindex=False).variance
        return self.__clean_prediction(prediction)

    def dump(self) -> str:
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        timestamp = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        filepath = os.path.join(self.model_directory, f"{timestamp}_{self.ticker}.pkl")
        joblib.dump(self.model, filepath)
        return filepath

    def load(self, path: str = None) -> None:
        if path:
            model_path = path
        else:
            pattern = os.path.join(self.model_directory, f"*{self.ticker}.pkl")
            files = sorted(glob(pattern))
            if not files:
                raise FileNotFoundError(f"No saved model found for '{self.ticker}'.")
            model_path = files[-1]
        self.model = joblib.load(model_path)

    def find_best_garch_params(self, p_range=range(1, 5), q_range=range(1, 5)):
        import itertools
        best_bic = float("inf")
        best_order = (1, 1)
        for p, q in itertools.product(p_range, q_range):
            try:
                model = arch_model(self.data, vol='GARCH', p=p, q=q, rescale=False)
                result = model.fit(disp='off')
                if result.bic < best_bic:
                    best_bic = result.bic
                    best_order = (p, q)
            except Exception:
                continue
        return (*best_order, best_bic)
