#  Human-level control through deep reinforcement learning
> Project for the course Capita Selecta Computer Science: Artificial Intelligence at KU Leuven

TODO: Description of project.

<!-- ![](header.png) -->

<!--
## Installation

OS X & Linux:

```sh
command
```
-->

## Usage example

TODO: A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

```sh
command
```  

## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
TODO: What things you need to install the software and how to install them

    dep 1, dep 2, ... TODO 

### Installing
TODO: A step by step series of examples that tell you have to get a development env running

Install the Python dependencies
```sh
pip3 install -r requirements.txt
```

### Data
Load the data into a running MongoDB instance 
```sh
mongorestore --drop -d cern_ldb --archive=data-cern/cern_ldb.mongodb.20170309.mki_v0_5.gz --gzip
```

The MongoDB dump archive was created as follows:
```sh
mongodump -d cern_ldb --archive=X.gz --gzip
```

## Built with
* [MongoDB](http://mongodb.com) - The database used
* [scikit-learn](http://scikit-learn.org) - Machine learning in Python
* ...

## Authors

* **Thiery Deruyterre** - [ThierryDeruyttere](https://github.com/ThierryDeruyttere)
* **Armin Halilovic** – [arminnh](http://github.com/arminnh/)


## License
Distributed under the MIT license. See ``LICENSE`` for more information.
