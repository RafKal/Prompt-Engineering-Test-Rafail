FROM nvcr.io/nvidia/pytorch:23.04-py3

# Arkansas specific workdir
WORKDIR /usr/src

# Update system
RUN apt update

# Upgrade pip \
RUN pip install --upgrade pip


# Install requirements
COPY ./requirements.dgx.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD python main.py