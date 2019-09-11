<template>
  <v-app id="inspire">
    <v-app-bar app color="indigo" dark>
      <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      <v-toolbar-title>GLIMVEC Viewer</v-toolbar-title>
    </v-app-bar>
    <v-content>
      <v-container fluid>
        <v-row align="center" v-for="(triple, index) in triples" :key="index" justify="center">
          <v-col xs="auto" sm="auto" md="auto" lg="auto" xl="auto">
            <v-btn
              @click="deleteTriple(index)"
              class="mx-2"
              outlined
              fab
              dark
              x-small
              color="primary"
            >
              <v-icon dark>mdi-minus</v-icon>
            </v-btn>
          </v-col>
          <v-col>
            <v-autocomplete label="Head" :items="entities" v-model="triple.head"></v-autocomplete>
          </v-col>
          <v-col>
            <v-autocomplete label="Relation" :items="relations" v-model="triple.relation"></v-autocomplete>
          </v-col>
          <v-col>
            <v-autocomplete label="Tail" :items="entities" v-model="triple.tail"></v-autocomplete>
          </v-col>
        </v-row>
        <v-btn @click="addTriple()" class="mx-2" outlined dark color="primary">
          <v-icon left dark>mdi-plus</v-icon>Add Triple
        </v-btn>
        <v-btn @click="query()" class="mx-2" dark color="primary">
          <v-icon left dark>mdi-database-search</v-icon>Query
        </v-btn>
        <v-row>
          <v-col>
            <v-data-table
              v-if="queried"
              :headers="headers"
              :items="similar_targets"
              :items-per-page="10"
              class="elevation-1"
            ></v-data-table>
          </v-col>
        </v-row>
      </v-container>
    </v-content>
    <v-footer color="indigo" app>
      <span class="white--text">Ryo Takahashi &copy; 2019</span>
    </v-footer>
  </v-app>
</template>

<script>
import axios from "axios";

export default {
  props: {
    source: String
  },

  data: () => ({
    drawer: null,
    triples: [{ head: "", relation: "", tail: "" }],
    entities: [],
    relations: [],
    headers: [
      { text: "Target word/phrase", value: "target" },
      { text: "Similarity", value: "similarity" }
    ],
    similar_targets: [],
    queried: false
  }),

  created() {
    this.getEntities();
    this.getRelations();
  },

  methods: {
    query() {
      this.queried = true;
      axios.post("http://localhost:5000/api/query", this.triples).then(resp => {
        this.similar_targets = resp.data;
      });
    },
    addTriple() {
      this.triples.push({
        head: "",
        relation: "",
        tail: ""
      });
    },
    deleteTriple(index) {
      this.triples.splice(index, 1);
    },
    getEntities() {
      axios.get("http://localhost:5000/api/entities").then(resp => {
        this.entities = resp.data;
      });
    },
    getRelations() {
      axios.get("http://localhost:5000/api/relations").then(resp => {
        this.relations = resp.data;
      });
    }
  }
};
</script>
