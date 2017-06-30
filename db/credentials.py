def python_postgresql():
    connect_str = "dbname='monkey_optim' user='monkey_optim' host='localhost' " + \
              "password='serrelab'"
    return connect_str


def postgresql_credentials():
    return {
            'username': 'monkey_optim',
            'password': 'serrelab'
           }


def postgresql_connection(port=''):
    unpw = postgresql_credentials()
    params = {
        'database': 'monkey_optim',
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    return params


def x7_credentials():
    return {
        'username': 'drew',
        'password': 'serrelab',
        'ssh_address': 'x7.clps.brown.edu'
       }
