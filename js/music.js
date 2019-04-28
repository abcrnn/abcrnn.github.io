async function loadDict() {
    await $.getJSON("test_model/model_dictionary.json", function(json) {
        console.log(json); // this will show the info it in firebug console
        obj = json;
    });
}

async function init() {
    document.getElementById('status').innerHTML = 'Loading model ...'
    console.log('Start loading model') 
    model = await tf.loadModel('test_model/model.json')
    console.log('Finish loading model') 
    document.getElementById('status').innerHTML = 'Model loaded!'
    $('#musicButton').css('display', 'inline-block');
}

$('#musicButton').on('click', async function(){
    $('#musicButton').css('display', 'none');
    $('.lds-ring').css('display', 'inline-block');
    console.log('Start loading dictionary')
    await loadDict();
    console.log('Finish loading dictionary')
    generate();
});

async function generate() {
    var seq_length = 200;
    await readFile(30);
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
    for (var i = 0; i < seq_length - seq_len; i++) {
        var batch = sequence_index.slice(Math.max(sequence_index.length - seq_len, 0))
        var predicted_probs = model.predictOnBatch(tf.tensor([batch]))

        predicted_probs.print()
        predicted_probs = tf.slice(predicted_probs, [0, 29, 0], [1, 1, 67])
        predicted_probs.print()
        sample = tf.argMax(predicted_probs.flatten())
        var {values, indices} = tf.topk(predicted_probs.flatten(), 3)
        values.print()
        indices.print()
        //predicted_probs.flatten().max().print();
        //predicted_probs.flatten().sum().print();
        sample.print()
        var char_note = obj.idx2char[sample.get()]
        console.log(char_note)
        if (char_note === '\n' && char_note === prev) {
            //continue;
            var {values, indices} = tf.topk(predicted_probs.flatten(), 2)
            console.log(indices.flatten().dataSync()[1])
            char_note = obj.idx2char[indices.flatten().dataSync()[1]]
            console.log(char_note)
            sequence_index.push(indices.flatten().dataSync()[1])
        } else {
            sequence_index.push(sample.get())
        }
        prev = char_note
    }
    var str_seq = "T: abcRNN\n";
    for (var i = 0; i < sequence_index.length; i++) {
        str_seq += obj.idx2char[sequence_index[i]];
    }           
    console.log(sequence_index)
    console.log(sequence_index.length)
    console.log(str_seq)
    $('.flow-text').css('display', 'none');
    $('.lds-ring').css('display', 'none');
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
    new ABCJS.Editor("abc", { paper_id: "paper0",
        generate_midi: true,
        midi_id:"midi",
        midi_download_id: "midi-download",
        generate_warnings: true,
        warnings_id:"warnings",
        abcjsParams: {
            generateDownload: true,
            //clickListener: selectionCallback
        }
    });
}

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

async function readFile(seq_len) {
    await jQuery.get("test_model/starting_dset.txt", function(textString) {
        //do what you want with the textString
        var start_idx = getRandomInt(textString.length - seq_len)
        starting_seed = textString.slice(start_idx, start_idx + seq_len)
    });
}
