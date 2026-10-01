106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
"""
                    except Exception:
                        # older xla versions or different runtime may not need explicit mark
                        pass
                except Exception:
                    logger.exception("TPU warmup iteration failed for %s:%s", self.name, self.version)

    def _to_device(self, x):
        # Map tensors to XLA device; if x is dict, move each tensor
        if isinstance(x, dict):
            return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in x.items()}
        if hasattr(x, "to"):
            return x.to(self.device)
        # if lists/tuples -> attempt tensor conversion
        try:
            return torch.tensor(x).to(self.device)
        except Exception:
            return x

    def _forward(self, inp):
        # model may accept dict or tensor
        if isinstance(inp, dict):
            try:
                return self._model(**inp)
            except TypeError:
                # fallback if model expects a single argument
                return self._model(list(inp.values()))
        else:
            return self._model(inp)

    def predict_batch(self, batched_inputs: List[Any]) -> List[Any]:
        """
        Run inference on a list of inputs. Each input converted/moved to device.
        Returns a list of outputs; tensors are moved to CPU and converted to python lists where possible.
        """
        if not self._loaded:
            self.load()
        outs = []
        with torch.no_grad():
            for inp in batched_inputs:
                try:
                    t = self._to_device(inp)
                    o = self._forward(t)
                    # ensure CPU numpy types for serialisation
                    if isinstance(o, torch.Tensor):
                        o = o.cpu().numpy().tolist()
                    outs.append(o)
                    # Force XLA step execution boundary
                    try:
                        xm.mark_step()
                    except Exception:
                        pass
                except Exception as exc:
                    logger.exception("TPU inference error: %s", exc)
                    outs.append({"error": str(exc)})
        return outs

    def cpu_offload(self):
        """
        Move model to CPU to free TPU memory. This requires reloading to XLA before next use.
        """
        if not self._loaded:
            return
        try:
            # best-effort: move module parameters to CPU
            try:
                self._model.to('cpu')
            except Exception:
                logger.exception("cpu_offload failed for TPU model")
            self._loaded = False
            logger.info("Offloaded TPU model %s:%s to CPU", self.name, self.version)
        except Exception:
            logger.exception("cpu_offload unexpected error")
api/model_loader_tpu.py
