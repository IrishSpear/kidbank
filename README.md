# KidBank

KidBank is a small, well-structured Python package for managing a reward-based
banking system for children. It models accounts, transactions, savings goals,
and rewards in an easy-to-read and fully tested code base.

## Features

- **Account management** – create accounts, deposit allowances, withdraw funds,
  and transfer money between siblings.
- **Reward tracking** – redeem rewards and keep a history of what was earned.
- **Savings goals** – define goals, contribute towards them, and monitor
  progress with percentage calculations.
- **Comprehensive statements** – generate readable account summaries that list
  recent activity, goals, and rewards.
- **Investing** – simulate portfolios with certificates of deposit that earn
  interest at an adjustable rate.
- **Test coverage** – unit tests exercise the core behaviours to keep the
  system reliable.

## Project structure

```
src/
  kidbank/
    __init__.py      # Public package exports
    account.py       # Core account logic and statement generation
    exceptions.py    # Custom exception hierarchy
    models.py        # Dataclasses for transactions, rewards, and goals
    money.py         # Helpers for working with monetary values
    service.py       # Facade for working with multiple accounts
pytest.ini           # Configures pytest to use the src layout
tests/
  test_account.py    # Behavioural tests for individual accounts
  test_service.py    # Integration tests for the KidBank service
```

## Getting started

1. Ensure you have Python 3.9 or newer available.
2. Install the development dependencies (only `pytest` is required) if you want
   to run the test suite.

## Usage example

```python
from kidbank import KidBank

bank = KidBank()
ava = bank.create_account("Ava", starting_balance=10)
bank.deposit("Ava", 5, description="Weekly chores")

bank.add_goal("Ava", name="Lego set", target_amount=40)
bank.contribute_to_goal("Ava", name="Lego set", amount=12.5)

bank.redeem_reward("Ava", name="Movie night", cost=5)

print(ava.generate_statement())
```

## Running the tests

```bash
python -m pytest
```

## Running the web frontend

The project now ships with a self-contained FastAPI frontend that persists data
in SQLite.  It reuses the KidBank domain model to provide parent and kid
dashboards, chores, allowances, goal tracking, rewards, CSV export, and the
stock-investing simulator described in the feature list.

1. Install the optional web dependencies:

   ```bash
   pip install fastapi uvicorn sqlmodel python-dotenv
   ```

2. (Optional) create a `.env` file with custom PINs and session secret:

   ```env
   MOM_PIN=1022
   DAD_PIN=2097
   SESSION_SECRET=replace-me
   ```

3. Launch the app:

   ```bash
   uvicorn kidbank.webapp:app --reload --host 0.0.0.0 --port 8000
   ```

Deploying into a Debian-based Proxmox LXC container only requires installing
the dependencies above, copying the repository, and starting Uvicorn.  The
SQLite database (`kidbank.db` by default) lives alongside the application and
will be created with all tables and migrations automatically on first run.
