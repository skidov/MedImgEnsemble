FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update &&  \
	apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 custom_user
RUN echo 'custom_user:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

ENTRYPOINT python main.py