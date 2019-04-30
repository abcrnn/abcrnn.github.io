async function loadDict(dict) {
    await $.getJSON("https://abcrnn.github.io/music_generation.python/music_model/" + modelName + "_music_model/model_dictionary.json", function(json) {
        obj = json;
    });
}

async function init(model_inp, id) {
    $('#musicButton').css('display', 'none');
    modelName = id;
    $('.lds-ellipsis').css('display', 'inline-block');    
    document.getElementById('musicButton').innerHTML = 'Let\'s Generate ' + id + ' Music! Click Me!'
    console.log('Start loading model') 
    model = await tf.loadModel(model_inp    )
    console.log('Finish loading model') 
    $('.lds-ellipsis').css('display', 'none'); 
    $('#musicButton').css('display', 'inline-block');
}

$('#musicButton').on('click', async function(){
    $('.lds-roller').css('display', 'inline-block'); 
    $('#musicButton').removeClass('pulse');
    console.log('Start loading dictionary')
    await loadDict();
    console.log('Finish loading dictionary')
    console.log(modelName)
    generate(300);
});

async function generate(seq_length) {
    var window_size = 30;
    await readFile(window_size);
    //var starting = "[A2F,2]dc [B2D2]A2 | [G2E,2]gf";
    //var starting = "[B,4d4] [c4A,4F,4] d4 | [dA,]^"
    var starting = starting_seed
    console.log(starting)
    var sequence_index = []
    console.log(obj)
    for (var i = 0; i < starting.length; i++) {
        console.log(starting.charAt(i));
        sequence_index.push(obj.char2idx[starting.charAt(i)])
    }
    console.log(sequence_index)
    var seq_len = starting.length
    var n_vocab = Object.keys(obj.idx2char).length;
    var prev = undefined
    var sample;
    for (var i = 0; i < seq_length - seq_len; i++) {
        var batch = sequence_index.slice(Math.max(sequence_index.length - seq_len, 0))
        var predicted_probs = model.predictOnBatch(tf.tensor([batch]))

        predicted_probs = tf.slice(predicted_probs, [0, window_size-1, 0], [1, 1, n_vocab]);//here 29: window size
        // sample = tf.argMax(predicted_probs.flatten()).get()//if chosen max prob notes
        console.log(tf.multinomial(tf.log(predicted_probs).squeeze(), 1).dataSync())
        sample  = tf.multinomial(tf.log(predicted_probs).squeeze(), 1).dataSync()[0];//make log prob: see api
        console.log(sample)
        console.log(tf.multinomial(tf.log(predicted_probs).squeeze(), 1).dataSync()[0])
        var {values, indices} = tf.topk(predicted_probs.flatten(), 3)
        values.print()
        indices.print()
        var char_note = obj.idx2char[sample]
        if (char_note === '\n' && char_note === prev) {
            //continue;
            var {values, indices} = tf.topk(predicted_probs.flatten(), 2)
            char_note = obj.idx2char[indices.flatten().dataSync()[1]]
            sequence_index.push(indices.flatten().dataSync()[1])
        } else {
            sequence_index.push(sample)
        }
        prev = char_note
    }
    var str_seq = "T: abcRNN\nM: 4/4\nK:Cmin\n";
    for (var i = 0; i < sequence_index.length; i++) {
        str_seq += obj.idx2char[sequence_index[i]];
    }           
    console.log(sequence_index)
    console.log(sequence_index.length)
    console.log(str_seq)
    $('.lds-roller').css('display', 'none'); 
    document.getElementById("abc").value = str_seq;
    initEditor();
    $('.abcjs').css('display', 'block');
}

function selectionCallback(abcelem) {
    var note = {};
    for (var key in abcelem) {
        if (abcelem.hasOwnProperty(key) && key !== "abselem")
            note[key] = abcelem[key];
    }
    console.log(abcelem);
    var el = document.getElementById("selection");
    el.innerHTML = "<b>selectionCallback parameter:</b><br>" + JSON.stringify(note);
}

function initEditor() {
    new ABCJS.Editor("abc", { canvas_id: "score-canvas",
        generate_midi: true,
        midi_id:"midi",
        midi_download_id: "midi-download",
        generate_warnings: true,
        warnings_id:"warnings",
        abcjsParams: {
            generateDownload: true,
            //clickListener: selectionCallback
        },
        inlineControls: {
            tempo: true,
        }
    });
}

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

async function readFile(seq_len) {
    await jQuery.get("https://abcrnn.github.io/music_generation.python/music_model/" + modelName + "_music_model/starting_seed_dbase.txt", function(textString) {
        //do what you want with the textString
        var start_idx = getRandomInt(textString.length - seq_len)
        starting_seed = textString.slice(start_idx, start_idx + seq_len)
    });
}
