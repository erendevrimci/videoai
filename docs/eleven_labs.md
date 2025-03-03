Create speech
POST


https://api.elevenlabs.io
/v1/text-to-speech/:voice_id
Convert text to speech using our library of over 3,000 voices across 32 languages.

Path parameters
voice_id
string
Required
ID of the voice to be used. Use the Get voices endpoint list all the available voices.

Query parameters
enable_logging
boolean
Optional
Defaults to true
When enable_logging is set to false zero retention mode will be used for the request. This will mean history features are unavailable for this request, including request stitching. Zero retention mode may only be used by enterprise customers.

optimize_streaming_latency
integer
Optional
Deprecated
You can turn on latency optimizations at some cost of quality. The best possible final latency varies by model. Possible values: 0 - default mode (no latency optimizations) 1 - normal latency optimizations (about 50% of possible latency improvement of option 3) 2 - strong latency optimizations (about 75% of possible latency improvement of option 3) 3 - max latency optimizations 4 - max latency optimizations, but also with text normalizer turned off for even more latency savings (best latency, but can mispronounce eg numbers and dates).

Defaults to None.

output_format
enum
Optional
The output format of the generated audio.


Show 11 enum values
Request
This endpoint expects an object.
text
string
Required
The text that will get converted into speech.

model_id
string
Optional
Defaults to eleven_monolingual_v1
Identifier of the model that will be used, you can query them using GET /v1/models. The model needs to have support for text to speech, you can check this using the can_do_text_to_speech property.

language_code
string
Optional
Language code (ISO 639-1) used to enforce a language for the model. Currently only Turbo v2.5 and Flash v2.5 support language enforcement. For other models, an error will be returned if language code is provided.

voice_settings
object
Optional
Voice settings overriding stored setttings for the given voice. They are applied only on the given request.


Hide 4 properties
stability
double
Optional
similarity_boost
double
Optional
style
double
Optional
Defaults to 0
use_speaker_boost
boolean
Optional
Defaults to true
pronunciation_dictionary_locators
list of objects
Optional
A list of pronunciation dictionary locators (id, version_id) to be applied to the text. They will be applied in order. You may have up to 3 locators per request


Show 2 properties
seed
integer
Optional
If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed. Must be integer between 0 and 4294967295.

previous_text
string
Optional
The text that came before the text of the current request. Can be used to improve the flow of prosody when concatenating together multiple generations or to influence the prosody in the current generation.

next_text
string
Optional
The text that comes after the text of the current request. Can be used to improve the flow of prosody when concatenating together multiple generations or to influence the prosody in the current generation.

previous_request_ids
list of strings
Optional
A list of request_id of the samples that were generated before this generation. Can be used to improve the flow of prosody when splitting up a large task into multiple requests. The results will be best when the same model is used across the generations. In case both previous_text and previous_request_ids is send, previous_text will be ignored. A maximum of 3 request_ids can be send.

next_request_ids
list of strings
Optional
A list of request_id of the samples that were generated before this generation. Can be used to improve the flow of prosody when splitting up a large task into multiple requests. The results will be best when the same model is used across the generations. In case both next_text and next_request_ids is send, next_text will be ignored. A maximum of 3 request_ids can be send.

use_pvc_as_ivc
boolean
Optional
Defaults to false
If true, we won’t use PVC version of the voice for the generation but the IVC version. This is a temporary workaround for higher latency in PVC versions.

apply_text_normalization
enum
Optional
Defaults to auto
Allowed values:
auto
on
off
This parameter controls text normalization with three modes: ‘auto’, ‘on’, and ‘off’. When set to ‘auto’, the system will automatically decide whether to apply text normalization (e.g., spelling out numbers). With ‘on’, text normalization will always be applied, while with ‘off’, it will be skipped. Cannot be turned on for ‘eleven_turbo_v2_5’ model.

from elevenlabs import ElevenLabs

client = ElevenLabs(
    api_key="YOUR_API_KEY",
)
client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    output_format="mp3_44100_128",
    text="The first move is what sets everything in motion.",
    model_id="eleven_multilingual_v2",
)
