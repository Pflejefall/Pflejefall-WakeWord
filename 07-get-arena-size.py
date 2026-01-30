import tensorflow as tf

def calculate_arena_size(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Summe der Größe aller Tensoren berechnen
    total_size = 0
    for detail in interpreter.get_tensor_details():
        # Die Größe in Bytes berechnen (Anzahl Elemente * Datentyp-Größe)
        size = detail['shape'].prod() * detail['dtype']().itemsize if len(detail['shape']) > 0 else 0
        total_size += size
    
    # TFLite Micro braucht einen Puffer für die Verwaltung (ca. 10-20% extra)
    # Ein Sicherheitsfaktor von 1.2 ist üblich
    recommended_arena = int(total_size * 1.2)
    
    print(f"Modell: {model_path}")
    print(f"Geschätzte Mindestgröße: {total_size} Bytes")
    print(f"Empfohlene 'tensor_arena_size' für JSON: {recommended_arena}")

calculate_arena_size("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")