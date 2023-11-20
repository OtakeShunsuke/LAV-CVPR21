FROM python:3.8.6

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN mkdir /work
# WORKDIR /work

# COPY BotCL .
COPY ./requirements.txt .

RUN apt update && apt install -y --no-install-recommends \
    libgl1-mesa-glx \
    zsh \
    tmux \
    sudo \
    && groupadd -g 1000 app_user \
    && useradd -m -s /bin/bash -u 1000 -g 1000 app_user \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt

USER app_user

RUN export GIT_SSL_NO_VERIFY=1 \
    && mkdir -p ${HOME}/dotfiles \
    && git clone https://github.com/OtakeShunsuke/dotfiles.git ${HOME}/dotfiles \
    && cd ${HOME}/dotfiles \
    && make install_locally


# ENTRYPOINT tmux new \; \
#     send-keys "zsh" Enter \; \
#     split-window -h  \; \
#     send-keys "zsh" Enter \; \
#     send-keys 'watch nvidia-smi' Enter \; \
#     split-window -v  \; \
#     new-window \; \
#     send-keys "zsh" Enter \; \
#     send-keys clear C-m \;nvcd src && 

WORKDIR /${HOME}

ENTRYPOINT tmux new \; \
    send-keys "cd ~/src && pip install --no-cache-dir -r requirements" Enter \; \
    send-keys 'cd' Enter \; \
    send-keys clear C-m \; \
    send-keys 'zsh' Enter \; \
    split-window -h  \; \
    send-keys "zsh" Enter \; \
    send-keys 'cd' Enter \; \
    send-keys 'watch nvidia-smi' Enter \; \
    split-window -v  \; \
    send-keys "zsh" Enter \; \
    send-keys 'cd' Enter \; \
    send-keys clear C-m \;



# COPY ./script.py .
# CMD python script.py

# CMD python main_recon.py --num_classes 10 --num_cpt 20 --lr 0.001 --epoch 50 --lr_drop 31
# CMD python vis_recon.py --num_classes 10 --num_cpt 20 --index 0 --top_sample 20 ---deactivate -1


# # # utils ----------------
# RUN apt update && apt install -y \
#     git \
#     tmux \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# COPY ./tmux.conf .

# # DL ----------------
# # RUN . ${DEFAULT_USER_HOME}/.zshrc && \
#     pip install \
#     setuptools \
#     jupyterlab \
#     widgetsnbextension && \
#     echo 'alias jl="DISPLAY=:0 jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &"' >> ${DEFAULT_USER_HOME}/.bash && \
#     echo 'alias tb="tensorboard --logdir runs --bind_all &"' >> ${DEFAULT_USER_HOME}/.bash


# RUN echo 'DISPLAY=:0 jupyter notebook --allow-root --ip=0.0.0.0 --port 8888 --notebook-dir=${DEFAULT_USER_HOME} --NotebookApp.password="" &' >> ${DEFAULT_USER_HOME}/run-jupyter.sh && \
#     chmod +x ${DEFAULT_USER_HOME}/run-jupyter.sh

# RUN sudo cp ${DEFAULT_USER_HOME}/.bash /root/.bash

# # auto start tmux and zsh
# ENTRYPOINT tmux new \; \
#             send-keys 'Xvfb :0 -screen 0 1400x900x24 & ' Enter \; \
# 	    send-keys 'x11vnc -display :0 -passwd 0123 -forever &' Enter \; \
#             split-window -v  \; \
#             send-keys "zsh" Enter \; \
#             send-keys "jupyter nbextension enable --py widgetsnbextension --sys-prefix" Enter \; \
#             send-keys "bash ${DEFAULT_USER_HOME}/run-jupyter.sh" Enter \; \
# 	   new-window \; \
#             send-keys "zsh" Enter \; \
#     	    send-keys clear C-m \;nvi