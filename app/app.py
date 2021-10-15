# -*- coding: utf-8 -*-

from flask import Flask, render_template, flash, request, Markup
import os
import argparse 
import re
from nlp_thes_augm.utils.common import Augmentation

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
'''
def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0].upper()
            hypernyms = read_fn(line_split[1])
            vocab[word].append(hypernyms)
    return vocab

def read_dataset_with_weight(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    word2weight = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0].upper()
            hypernyms = read_fn(line_split[1])
            weight = line_split[2]
            vocab[word].append(hypernyms)
            word2weight[word].append(float(f'{weight:.5}'))
    return vocab, word2weight

class Predict():
    def __init__(self, name, score, rel='miss'):
        self.name = name
        self.score = score
        self.rel = rel



# Never render this form publicly because it won't have a csrf_token
selector_choices = [('miss', 'Пусто'), ('issynonym', 'Синоним'), ('ishypernym','Гипероним'),('ishyponym','Гипоним'),('ispart','Часть'),('iswhole','Целое')]
class PredictionForm(Form):
    name = StringField('name', render_kw={'readonly': True})
    score = StringField('score', render_kw={'readonly': True})
    index = IntegerField('index', render_kw={'readonly': True})
    rels = SelectField(
        'Label', 
        choices=selector_choices, 
        default='miss'
    )
    #id = fields.IntegerField(validators=[validators.required()], widget=HiddenInput())
    #home_score = fields.TextField(validators=[validators.required()])
    #away_score = fields.TextField(validators=[validators.required()])

class ReusableForm(Form):
    predicts = FieldList(FormField(PredictionForm))
    word = StringField('Word')

class HypernymMarkUp():
    methods = ['GET', 'POST']
    def __init__(self, predict_path, thes_path):
        self.thesaurus = RuThes(thes_path)

        if os.path.exists('markup_test_dump'):
            self.predicts, self.curr_word_idx = pickle.load(codecs.open('markup_test_dump', 'rb'))
        else:
            predicts, word2weight = read_dataset_with_weight(predict_path)
            self.predicts = []
            for word in predicts:
                word_preds = []
                for concept, weight in zip(predicts[word], word2weight[word]):
                    word_preds.append(Predict(self.thesaurus.synsets[int(concept)].synset_name, weight))

                max_weight = max(word2weight[word])
                self.predicts.append([word, word_preds, max_weight])

            self.predicts = sorted(self.predicts, key=lambda x: -x[2])
            self.curr_word_idx = 0

        self.name2id = {}
        for synset_id in self.thesaurus.synsets:
            synset_name = self.thesaurus.synsets[synset_id].synset_name
            if synset_name in self.name2id:
                print(synset_name)
            self.name2id[synset_name] = synset_id
 
    def save(self):
        pickle.dump([self.predicts, self.curr_word_idx], codecs.open('markup_test_dump', 'wb'))

    def save_table(self):
        print('SAVE')
        predicts = request.get_json()
        print(predicts)
        for i, pred in enumerate(predicts):
            if i < len(self.predicts[self.curr_word_idx][1]):
                self.predicts[self.curr_word_idx][1][i].rel = pred['rel']
            else:
                self.predicts[self.curr_word_idx][1].append(Predict(pred['name'], pred['score'], pred['rel']))
        self.save()
        return self.render_page()

    def next(self):
        print('NEXT')
        self.curr_word_idx += 1
        return self.render_page()

    def prev(self):
        print('PREV')
        self.curr_word_idx -= 1
        return self.render_page()

    def render_page(self):
        print('render_page')
        curr_word_preds = self.predicts[self.curr_word_idx]
    
        form = ReusableForm(word=curr_word_preds[0], predicts=[dict({'name': pred.name, 'score': pred.score}) for pred in curr_word_preds[1]])
        form.word = curr_word_preds[0]
        for i in range(len(curr_word_preds[1])):
            form.predicts[i].index = i
            form.predicts[i].name = curr_word_preds[1][i].name
            form.predicts[i].score = curr_word_preds[1][i].score
            form.predicts[i].rels.process_data(curr_word_preds[1][i].rel)

        nodes = self.get_nodes_notfar(curr_word_preds[1], distance=2)
        hypernym_map = self.get_rel_map(nodes, 'hypernym')
        hyponym_map = self.get_rel_map(nodes, 'hyponym')

        #print(form.data)
        return render_template('hello.html', form=form, hypernym_map=hypernym_map, hyponym_map=hyponym_map, selector_choices=selector_choices)

    def hello(self):
        print('HELLO')
        print(request.method)
        if request.method == 'POST':
            if 'prev' in request.form:
                return self.prev()
            if 'next' in request.form:
                return self.next()

        return self.render_page()
        
    def get_nodes_notfar(self, word_preds, distance):
        nodes = set([pred.name for pred in word_preds])
        for pred in word_preds:
            synset_id = self.name2id[pred.name]
            nodes_ids = self._get_nodes_notfar(synset_id, distance)
            nodes.update([self.thesaurus.synsets[synset_id].synset_name for synset_id in nodes_ids])

        return nodes

    def _get_nodes_notfar(self, synset_id, distance):
        if distance == 0:
            return []
        synset = self.thesaurus.synsets[synset_id]
        hypernyms = [s.synset_id for s in synset.rels.get('hypernym', [])]
        hyponyms = [s.synset_id for s in synset.rels.get('hyponym', [])]

        synsets_ids = set(hypernyms + hyponyms)
        for neib_synset_id in synsets_ids.copy():
            synsets_ids.update(self._get_nodes_notfar(neib_synset_id, distance - 1))

        return synsets_ids

    def get_rel_map(self, nodes, rel):
        rel_map = {}
        for synset_name in nodes:
            synset_id = self.name2id[synset_name]
            synset = self.thesaurus.synsets[synset_id]
            rels = synset.rels.get(rel, [])
            if len(rels) > 0:
                rel_map[synset_name] = [s.synset_name for s in rels]

        return rel_map

'''
from nlp_thes_augm.models.augmentation import WordnetAugmentator
from nlp_thes_augm.utils.wordnet import RuWordNet
from nlp_thes_augm.utils.common import inject_concept_tokens

class AugmentationApp:
    def __init__(self, augmentator):
        self.augmentator = augmentator

    def get_prediction(self):
        max_concept_count = 5
        augmentations_count = 500
        topk = 5

        if request.method == 'POST':
            print(request.form)
            max_concept_count = int(request.form['ccount'])
            augmentations_count = int(request.form['afcount'])
            topk = int(request.form['arcount'])

            text = request.form['text']
            text = ' '.join([w.strip() for w in text.split()])
            augmentations, detected_concepts = augmentator.augmentate(
                text, augmentations_count=augmentations_count, topk=topk, max_concept_count=max_concept_count, bs=100
            )
            detected_concepts_injected = inject_concept_tokens(detected_concepts, text)
            detected_concepts_injected = self._markup_detected_concepts(detected_concepts_injected)
            augmentation_table = self._generate_table(augmentations)
            return render_template('home.html', text=text, markup=detected_concepts_injected, augmentation_table=augmentation_table, ccount=max_concept_count, afcount=augmentations_count, arcount=topk)

        return render_template('home.html', ccount=max_concept_count, afcount=augmentations_count, arcount=topk)

    def _markup_detected_concepts(self, detected_concepts_injected):
        words = detected_concepts_injected.split()
        _words = []
        for w in words:
            if '{{' in w and '}}' in w:
                w_raw, w_norm, concepts = re.findall(r'{{(.*?)}}', w)[0].split('|')
                concepts = concepts.split(',')
                concepts = '<br>'.join([self.augmentator.wordnet.synsets[c].synset_name for c in concepts])
                hover_text = f'<b>Нормальная форма:</b><br>{w_norm}<br><b>Концепты:</b><br>{concepts}'
                w = f'<a href=\"#\" data-toggle=\"tooltip\" data-html=\"true\" title=\"{hover_text}\">' + w_raw + '</a>' 
            _words.append(w)
        return Markup(' '.join(_words))

    def _markup_augmentation(self, augm):
        augm_text_tokenized = re.findall(self.augmentator.text_preprocessor.sep_reg, augm.restored_augmentation)
        augm_concepts = augm.concepts
        augm_senses = augm.senses
        augm_synsets_ids = augm.synsets_ids

        current_concept_idx = 0
        current_concept_num = augm_concepts[current_concept_idx].tokens[0].num
        shift = 0
        counter = 0

        result_markup_text = ''
        for i, token in enumerate(augm_text_tokenized):
            if counter > 0:
                result_markup_text += ' ' + token
                if counter == 1:
                    result_markup_text += '</a>'
                counter -= 1
                continue
            if i == current_concept_num + shift:

                selected_synset_name = self.augmentator.wordnet.synsets[augm_synsets_ids[current_concept_idx]].synset_name
                selected_sense =  augm_senses[current_concept_idx]

                hover_text = f'<b>Выбранный концепт:</b><br>{selected_synset_name}<br><b>Выбранный текстовый вход:</b><br>{selected_sense}'
                open_tag = f'<a href=\"#\" data-toggle=\"tooltip\" data-html=\"true\" title=\"{hover_text}\">'
                result_markup_text += open_tag

                counter = len(augm_senses[current_concept_idx].split('_'))
                shift += counter - len(augm_concepts[current_concept_idx].tokens)
                current_concept_idx += 1
                current_concept_num = augm_concepts[current_concept_idx].tokens[0].num if current_concept_idx < len(augm_concepts) else len(augm_text_tokenized)

            result_markup_text += ' ' + token
            if counter == 1:
                result_markup_text += '</a>'
            counter -= 1
        return result_markup_text.strip()

    def _generate_table(self, augmentations):
        table = '<table class=\"info\" border=1 frame=void rules=rows>'
        for augm in augmentations:
            tr = '<tr>'

            try:
                td_augm = '<td width=\"100%\">' + self._markup_augmentation(augm)+ '</td>'
            except:
                td_augm = '<td width=\"100%\">' + augm.restored_augmentation + '</td>'
            td_score = '<td width=\"10%\">' + f'{augm.score:.3f}' + '</td>'

            tr += td_augm + td_score
            tr += '</tr>'
            table += tr

        table += '</table>'
        return Markup(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordnet_path', type=str)
    parser.add_argument('--t5_model_path', type=str)
    parser.add_argument('--gpt2_model_path', type=str)
    args = parser.parse_args()

    wordnet = RuWordNet(args.wordnet_path)
    augmentator = WordnetAugmentator(wordnet, args.t5_model_path, args.gpt2_model_path)
    augmentation_app = AugmentationApp(augmentator)
    #hypernym_markup = HypernymMarkUp(args.predict_path, args.thes_path)

    app.add_url_rule('/', view_func=augmentation_app.get_prediction, methods=['POST', 'GET'])
    #app.add_url_rule('/save_table', view_func=hypernym_markup.save_table, methods=['POST'])
    #app.add_url_rule('/next', view_func=hypernym_markup.next, methods=['POST', 'GET'])
    #app.add_url_rule('/prev', view_func=hypernym_markup.prev, methods=['POST', 'GET'])

    app.run(debug=False, host='127.0.0.1', port=5000)
