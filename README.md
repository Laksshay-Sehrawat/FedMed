# Federated Learning with Flower

BTP repo by Laksshay, Vanshika, Saumya

## Prerequisites

- Python 3.7 or higher
- `pip` package manager

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com:Laksshay-Sehrawat/FedMed.git
   ```

## Virtual env
  ```sh
  python -m venv env
  source env/bin/activate
  ```

## Install requirements
  ```sh
  pip install -r requirements.txt
  ```

## Running the Server

  Start the Flower server:
  ```sh
    python server.py
  ```
  Running the Clients -> Open two terminal windows.

  In the first terminal, run the first client:
  ```sh
    python client.py ./data/client1
  ```

  In the second terminal, run the second client:
  ```sh
    python client.py ./data/client2
  ```


