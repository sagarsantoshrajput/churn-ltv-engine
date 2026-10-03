"""Week 3: score every customer in the warehouse and publish to `churn_predictions`."""
import logging
from datetime import datetime

from src import db
from src.inference import ScoringService

log = logging.getLogger(__name__)


def run():
    svc = ScoringService()
    df = db.load_customers()
    out = svc.score(df.drop(columns=["churn"]))
    out["scored_at"] = datetime.utcnow()
    db.replace_rows(db.churn_predictions, out)
    log.info("Scored %d customers. Risk mix: %s", len(out), out["risk_band"].value_counts().to_dict())
    log.info("Priority mix: %s", out["retention_priority"].value_counts().to_dict())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
