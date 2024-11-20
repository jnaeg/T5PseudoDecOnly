#pushes the trained models to the hub

# ... train your model on some task and dataset
# ... save it in "path/to/awesome-name-you-picked"
# ... huggingface-cli login

import transformers
from modeling_t5dec import T5PseudoDecForConditionalGeneration

#push pt model to hub
pt_T5Dec = T5PseudoDecForConditionalGeneration.from_pretrained("path/to/awesome-name-you-picked") 

#convert to tensorflow, push to hub
tf_T5Dec = T5PseudoDecForConditionalGeneration.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
tf_T5Dec.save_pretrained("path/to/awesome-name-you-picked") 
tf_T5Dec.push_to_hub("T5PseudoDecoderOnly")
