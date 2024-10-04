import json
import argparse

fake_user: json = [
    {
     "user": "eril",
     "password": "$2b$12$LPsngZCPSlXdD4GLMpzvZe5qT7ZIHtUkMkWgDi/rsLcdfN59zPEDO",
     "full_name": "Eril Sanjaya",
     "email": "erilsanjaya@gmail.com",
     "disabled": False
     },

    {"user": "admin",
     "password": "$2b$12$C2Q8g.REixl237JQAuGwIeNfTdHmmBxmRSIh7ARgxp7OB92F3p3mi",
     "full_name": "Admin",
     "email": "admin@gmail.com",
     "disabled": False
     }
]


if __name__ == "__main__":

    print(fake_user)
