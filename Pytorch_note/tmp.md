$$𝑧=𝑤_{0}𝑥_{0}+𝑤_{1}𝑥_{1}+…+𝑤_{n}𝑥_{n}+𝑏$$

$$\frac{-lnL(w,b)}{\partial{w_{i}}} = \sum_{n}-[\hat{y} \frac{lnf_{w,b}(x^{n})}{\partial{w_{i}}} + (1-\hat{y}^{n})\frac{ln(1-f_{w,b}(x^{n}))}{\partial{w_{i}}} ]$$

$$\frac{lnf_{w,b}(x^{n})}{\partial{w_{i}}} = \frac{lnf_{w,b}(x^{n})}{\partial{z}} \frac{\partial{z}}{\partial{w_{i}}}$$

$$\frac{\partial{z}}{\partial{w_{i}}} = x_{i}$$

$$\frac{-lnL(w,b)}{\partial{w_{i}}} = \sum_{n}-[\hat{y}^{n}(1-f_{w,b}(x^{n}_{i}))x_{i}^{n} - (1-\hat{y}^{n})f_{w,b}(x^{n})x_{i}^{n}]$$

$$=\sum_{n}-(\hat{y}^{n} - f_{w,b}(x^{n}))x_{i}^{n}$$
$$L(f) = \frac{1}{2}\sum_{n}(f_{w,b}(x^{n}) - \hat{y}^{n})^{2}$$

$$\frac{\partial(f_{w,b}(x) - \hat{y})^{2} }{\partial{w_{i}}}  = 2(f_{w,b} - \hat{y}) \frac{\partial{f_{w,b}(x)}}{\partial{z}} \frac{\partial{z}}{\partial{w_{i}}} $$

$$= 2(f_{w,b} - \hat{y})f_{w,b}(x)(1-f_{w,b})x_{i}$$