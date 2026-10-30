"""Pure command builder for the qualified dual-B70 Strata profile."""
import ipaddress
from ai_toolbox_cockpit.runtime.toolboxes import upgrade_groups_for_podman
from .model_manager import model_root, prepared

CONTAINER_NAME='strata-cockpit-server'


def build_server_cmd(*,engine,image,engine_args,platform_id,model,directory,
                     host='127.0.0.1',port=8000,context_size=68096,speculation='mtp'):
    if platform_id!='intel-b70':raise ValueError('This Strata profile requires two Intel B70 GPUs')
    if engine not in {'podman','docker'}:raise ValueError('Select Podman or Docker')
    if not image or image.startswith('-'):raise ValueError('Select a Strata toolbox image')
    if type(context_size) is not int or not 2048<=context_size<=68096:raise ValueError('Context must be between 2048 and 68096')
    if type(port) is not int or not 1<=port<=65535:raise ValueError('Invalid API port')
    address=ipaddress.ip_address('127.0.0.1' if host=='localhost' else host)
    if speculation not in {'baseline','mtp'}:raise ValueError('Select target-only or MTP3')
    if not prepared(model,directory,mtp=speculation=='mtp'):raise ValueError('Download and prepare the Strata model in Models first')
    root=model_root(model,directory)
    bind=f'[{address}]' if address.version==6 else str(address)
    cmd=[engine,'run','--rm','-it','--name',CONTAINER_NAME]
    cmd+=upgrade_groups_for_podman(engine,list(engine_args))
    cmd+=['--ipc=host']
    if engine=='podman':cmd+=['--security-opt','label=disable','--userns=keep-id']
    cmd+=['-e','HOME=/tmp','-p',f'{bind}:{port}:{port}','-v',f'{root}:/models:ro',
          image,'strata-serve','--model',f'/models/{model.files[0][0]}','--pack','/models/pack',
          '--context',str(context_size),'--host','0.0.0.0','--port',str(port)]
    if speculation=='mtp':cmd+=['--mtp','/models/mtp/rt']
    return cmd
